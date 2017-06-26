import tensorflow as tf
import pickle_utils as pu
import itertools as it
import numpy as np
import os.path
import os
import sys
import sklearn.metrics
import subprocess
from tqdm import tqdm

from read_tfrecords import build_input_machinery
from bb_alpha_inputs import build_sampler
import model

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('command', 'train', 'What to do [train, validate, profile_training]')
    flags.DEFINE_string('load_file', '', 'What file to load for validating?')
    flags.DEFINE_string('log_dir', None, 'Base directory for logs')
    flags.DEFINE_integer('min_after_dequeue', 5000,
                        'Minimum number of examples to draw from in the RandomShuffleQueue')
    flags.DEFINE_integer('n_queue_threads', 2, 'number of queue threads')
    flags.DEFINE_integer('batch_size', 64, 'batch size for training')
    flags.DEFINE_integer('num_epochs', 1000, 'number of training epochs')
    flags.DEFINE_integer('hidden_units', 100, 'Number of hidden units per LSTM layer')
    flags.DEFINE_integer('hidden_layers', 1, 'Number of hidden LSTM layers')
    flags.DEFINE_integer('num_samples', 10,
                        'Number of samples for the monte-carlo imputation')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate for ADAM')
    flags.DEFINE_float('dropout', 0.5, 'probability of keeping a neuron on')
    flags.DEFINE_boolean('layer_norm', True, 'Whether to use Layer Normalisation')
    flags.DEFINE_string('model', 'GRUD', 'the model to use')
    flags.DEFINE_string('dataset', '../clean_ventilation/dataset', 'Dataset folder')
    flags.DEFINE_string('interpolation', '../clean_ventilation/interpolation',
                        'Interpolation data/trained model folder')
    del flags
FLAGS = tf.app.flags.FLAGS

def build_validation_summaries(hours_before, metric):
    summaries_ph = {}
    summaries = []
    for name in 'training', 'validation':
        summaries_ph[name] = []
        with tf.variable_scope(name):
            for h in hours_before:
                ph = tf.placeholder(dtype=tf.float32, shape=[],
                                    name="{:s}/{:d}_hours".format(metric, h))
                s = tf.summary.scalar('{:s}/{:d}_hours'.format(metric, h), ph)
                summaries.append(s)
                summaries_ph[name].append(ph)
    validation_summary = tf.summary.merge(summaries)
    return summaries_ph, validation_summary

def validate_checkpoint(persist):
    if 'latest_ckpt' not in persist:
        persist['latest_ckpt'] = None
        persist['running_procs'] = []
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.log_dir)
    if latest_ckpt is None or latest_ckpt == persist['latest_ckpt']:
        return
    persist['latest_ckpt'] = latest_ckpt

    args = [sys.executable, sys.argv[0]]
    flags = FLAGS.__flags.copy()
    flags['command'] = 'validate'
    flags['num_epochs'] = 1
    flags['load_file'] = latest_ckpt
    for name, value in flags.items():
        args.append("--{:s}={}".format(name, value))

    for proc in persist['running_procs'][:]:
        if proc.poll() is not None:
            persist['running_procs'].remove(proc)
            print("An earlier subprocess terminated")
    print("Calling subprocess", args)
    persist['running_procs'].append(subprocess.Popen(args))

def main(_):
    assert FLAGS.log_dir is not None
    config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.n_threads)

    shuffle = True
    feature_numbers = pu.load(os.path.join(FLAGS.dataset, 'feature_numbers.pkl.gz'))
    print(feature_numbers)
    if FLAGS.command == 'validate':
        assert FLAGS.load_file, "Must indicate a file to load"
        assert os.path.isfile(FLAGS.load_file+'.index')
        assert os.path.dirname(FLAGS.load_file) == FLAGS.log_dir.rstrip("/"), \
            "File to load must be in log_dir"
        assert FLAGS.num_epochs == 1, "We only want to go through the sets once"
        shuffle = False
        validation = build_input_machinery([os.path.join(FLAGS.dataset,
                                                         "validation_0.tfrecords")],
                                           feature_numbers, False,
                                           FLAGS.num_epochs, FLAGS.batch_size,
                                           FLAGS.min_after_dequeue,
                                           FLAGS.n_queue_threads)

    training = build_input_machinery([os.path.join(FLAGS.dataset,
                                                   "train_0.tfrecords")],
                                     feature_numbers, shuffle,
                                     FLAGS.num_epochs, FLAGS.batch_size,
                                     FLAGS.min_after_dequeue,
                                     FLAGS.n_queue_threads)

    def ds_load(f, flag='dataset'):
        total_num_features = (feature_numbers['numerical_ts'] +
                              feature_numbers['categorical_ts'])
        d = getattr(FLAGS, flag)
        return pu.load(os.path.join(d, f.format(total_num_features)))
    input_means_numerical = pu.load(os.path.join(FLAGS.dataset, 'means.pkl.gz'))
    input_means_numerical = input_means_numerical[0] / input_means_numerical[1]
    # The number of categories is computed from _all_ the data set, since we
    # need to build our model to fit enough of them in its arrays. However it
    # is _not overfitting_, even if one of the categories does not appear in
    # the training set. If that is the case, we will never train that
    # category's embeddings although we do know it exists.
    number_of_categories = ds_load('number_of_categories_{:d}.pkl.gz')
    _, _, _, numerical_headers, categorical_headers, treatments_headers = \
        ds_load('headers_{:d}.pkl.gz')
    del _

    def build_model(inputs, reuse=None):
        Model = getattr(model, FLAGS.model)
        with tf.variable_scope(FLAGS.model, reuse=reuse):
            m = Model(num_units=FLAGS.hidden_units,
                      num_layers=FLAGS.hidden_layers,
                      inputs_dict=inputs,
                      input_means_numerical=input_means_numerical,
                      number_of_categories=number_of_categories,
                      categorical_headers=categorical_headers,
                      numerical_headers=numerical_headers,
                      default_batch_size=FLAGS.batch_size,
                      layer_norm=FLAGS.layer_norm,
                      interpolation_dir=FLAGS.interpolation,
                      num_samples=FLAGS.num_samples)
            return m

    m = build_model(training)

    if FLAGS.command == 'validate':
        v_m = build_model(validation, reuse=True)
        hours_before = [24, 12, 8, 4, 0]
        with tf.variable_scope('metrics'):
            rs_raw = model.ventilation_risk_score_raw(training, m)
            v_rs_raw = model.ventilation_risk_score_raw(validation, v_m)
            tpr_ppv_ph_d, tpr_ppv_summary = \
               build_validation_summaries(hours_before, 'tpr_ppv')
            roc_ph_d, roc_summary = \
               build_validation_summaries(hours_before, 'roc')

    global_step = tf.train.get_or_create_global_step()
    train_op = (tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
                .minimize(m['loss'], global_step=global_step))

    saver = tf.train.Saver(max_to_keep=0)
    with tf.variable_scope('summaries'):
        loss_ph = tf.placeholder(shape=[], dtype=tf.float32)
        training_summary = tf.summary.scalar('training/loss', loss_ph)

    if FLAGS.command == 'train':
        def load_num_model(i):
            num_dir = os.path.join(FLAGS.interpolation, 'trained',
                                    'num_{:d}'.format(i))
            num_ckpt = pu.load(os.path.join(num_dir, 'validated_best.pkl'))
            ckpt_fname = os.path.join(num_dir, 'ckpt-{:d}'.format(num_ckpt))
            saver_d = {}
            with tf.variable_scope("", reuse=True):
                for var_name, _ in tf.contrib.framework.list_variables(ckpt_fname):
                    if var_name.startswith("num_") and 'Adam' not in var_name:
                        saver_d[var_name] = tf.get_variable("BayesDropout/rnn/bayes_dropout_cell/num_inputs/"+var_name)
            _saver = tf.train.Saver(saver_d)
            return (_saver, ckpt_fname)

        if FLAGS.model == 'GRUD' or FLAGS.impute_as_zeros:
            num_savers = []
        else:
            num_savers = list(map(load_num_model,
                              range(feature_numbers['numerical_ts'])))

        sv = tf.train.Supervisor(is_chief=True,
                                logdir=FLAGS.log_dir,
                                summary_op=None,
                                saver=saver,
                                global_step=global_step,
                                save_model_secs=600)

        # Save invocation flags to log_dir
        with open(os.path.join(FLAGS.log_dir, 'flags.txt'), 'w') as f:
            f.write(repr(FLAGS.__flags)+'\n')

        normal_ops = [train_op, m['loss']]
        summary_ops = [train_op, m['loss'], training_summary]
        d = {m['keep_prob']: FLAGS.dropout, loss_ph: 0.0}

        validate_checkpoint_persist = {}
        with sv.managed_session(config=config) as sess:
            sv.loop(300, validate_checkpoint, args=(validate_checkpoint_persist,))
            sess.run(tf.get_collection('make_embeddings_nan'))
            for _saver, ckpt_fname in num_savers:
                _saver.restore(sess, ckpt_fname)

            for step in tqdm(it.count(1)):
                if sv.should_stop():
                    break
                if step % 100 == 0:
                    d[loss_ph] /= 100
                    _, d[loss_ph], summ = sess.run(summary_ops, d)
                    sv.summary_computed(sess, summ)
                else:
                    _, loss = sess.run(normal_ops, d)
                    d[loss_ph] += loss
            sv.stop()

    elif FLAGS.command == 'validate':
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            queue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver.restore(sess, FLAGS.load_file)
            sess.run(tf.get_collection('make_embeddings_nan'))

            d = {m['keep_prob']: 1.0, v_m['keep_prob']: 1.0}
            labels_preds = {}
            for n in 'training', 'validation':
                labels_preds[n] = list(([], []) for _ in hours_before)

            def extend_labels_preds(tensors, name):
                try:
                    for step in range(5):
                        if coord.should_stop():
                            break
                        results = sess.run(tensors, d)
                        risk_scores = model.ventilation_risk_score(*results, hours_before)
                        for i, h in enumerate(hours_before):
                            labels_preds[name][i][0].extend(risk_scores[h][0])
                            labels_preds[name][i][1].extend(risk_scores[h][1])
                        print("Done {:d}th {:s} evaluation step".format(step, name))
                except tf.errors.OutOfRangeError:
                    pass

            extend_labels_preds(rs_raw, 'training')
            extend_labels_preds(v_rs_raw, 'validation')

            coord.request_stop()
            coord.join(queue_threads)

            summary_writer = tf.summary.FileWriter(FLAGS.log_dir)

            feed_dict = {}
            for name in labels_preds:
                assert len(hours_before) == len(labels_preds[name])
                assert len(hours_before) == len(roc_ph_d[name])
                assert len(hours_before) == len(tpr_ppv_ph_d[name])
                for hour, tpr_ppv_ph, roc_ph, lps in zip(hours_before,
                        tpr_ppv_ph_d[name], roc_ph_d[name], labels_preds[name]):
                    if len(list(filter(lambda i: i>0.1, lps[0]))) == 0:
                        import pdb
                        pdb.set_trace()
                    feed_dict[tpr_ppv_ph] = model.binary_auc_tpr_ppv(*lps)
                    feed_dict[roc_ph] = sklearn.metrics.roc_auc_score(
                        *lps, average='macro')
                    print("{:d} hours: {} TPR/PPV, {} ROC".format(
                        hour, feed_dict[tpr_ppv_ph], feed_dict[roc_ph]))
            s1, s2, t = sess.run([tpr_ppv_summary, roc_summary, global_step], feed_dict)
            summary_writer.add_summary(s1, t)
            summary_writer.add_summary(s2, t)

    else:
        raise ValueError("Unknown command: `{:s}`".format(FLAGS.command))

if __name__ == '__main__':
    tf.app.run()
