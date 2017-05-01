    fillna = dict(it.chain(
        map(lambda e: (e, 0.0), zero_headers),
        #map(lambda e: (e, False),
        #    list(filter(lambda h: h[1:7] != ' pred ', bool_headers))),
#        map(lambda e: (e, 'bfill'),
#            list(filter(lambda h: h[1:7] == ' pred ', bool_headers))),
    ))

    df = pd.read_csv('../mimic.csv.gz', nrows=10000,
                     index_col=False, engine='c',
                     true_values=[b'1'], false_values=[b'0'],
                     usecols=usecols,dtype=dtype).fillna(fillna)

    observations = df.filter(regex=r'^.(?! pred ).*$').fillna(method='ffill').fillna(method='bfill').dropna(axis='columns')
    labels = df.filter(regex=r'^. pred .*$')
    ventilation_ends = np.argwhere(pd.notnull(labels.iloc[:,0]))

    icustays_start = {}
    for i, v in enumerate(observations['icustay_id']):
        if v not in icustays_start:
            icustays_start[v] = i
    start_icustays = sorted(list(map(lambda t: (t[1],t[0]), icustays_start.items())))

    labels_notime = []
    icustay_labels = collections.defaultdict(lambda: collections.defaultdict(lambda: [], {}), {})
    sicu_i = -1
    prev_icustay = None
    targets = []
    data = observations.iloc[:0,:]
    for v_end in map(int, ventilation_ends):
        # Find which ICU stay starts just before the ventilator end
        while start_icustays[sicu_i+1][0] <= v_end:
            sicu_i += 1
        icustay_id = start_icustays[sicu_i][1]
        if icustay_id != prev_icustay:
            prev_icustay = icustay_id
            n_vent = -1
        n_vent += 1
        v_end_hour = int(observations['hour'][v_end])
        cut_data = observations.iloc[start_icustays[sicu_i][0]:v_end,:]
        cut_data = cut_data[cut_data.hour >= -48]
        for _i, hours_before in enumerate([4, 8, 12, 24]):
            id = icustay_id*1000 + n_vent*10 + _i
            icustay_labels[icustay_id][v_end_hour].append(id)
            targets.append((id, labels.iloc[v_end,0], labels.iloc[v_end,1]))
            vent_df = cut_data[cut_data.hour <= v_end_hour-hours_before]
            prev_len = data.shape[0]
            data = pd.concat([data, vent_df])
            data.iloc[prev_len:,0] = id

    ids, vent_is_last, hours_to_death = zip(*targets)
    y = pd.Series(data=vent_is_last, index=ids)

    #tsfresh.feature_selection.FeatureSignificanceTestsSettings.n_processes = 8
    settings = tsfresh.feature_extraction.ReasonableFeatureExtractionSettings()
    settings.n_processes = multiprocessing.cpu_count()
    X = tsfresh.feature_extraction.extraction.extract_features(
        data, column_id='icustay_id', column_sort='hour',
        feature_extraction_settings=settings,
        parallelization='per_kind')
    #X = tsfresh.extract_relevant_features(data, y, column_id='icustay_id', column_sort='hour')

