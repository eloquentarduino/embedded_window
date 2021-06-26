import numpy as np


class Window:
    def __init__(self, length, shift=1, agreement=0.5):
        """

        :param length:
        :param shift:
        :param agreement:
        """
        assert length > 1, 'length MUST be greater than 1'
        assert shift > 0, 'shift MUST be greater than 0'
        assert agreement > 0, 'agreement MUST be greater than 0'

        self.length = length
        self.shift = shift if shift >= 1 else shift * length
        self.agreement = agreement if agreement >= 1 else agreement * length
        self.num_features = None

    def fit_transform(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self.num_features = X.shape[1]

        return self.transform(X, y)

    def transform(self, X, y):
        """
        Fit window to X, y
        :param X:
        :param y:
        :return:
        """
        num_samples = len(X)
        w = np.arange(self.length)
        t = np.arange(num_samples - self.length + 1)
        idx = (w + t.reshape((-1, 1)))[::self.shift]

        X_window = X[idx]
        y_window = y[idx]
        mask = np.asarray([self._mode(yi) for yi in y_window], dtype=int)

        X_window = X_window[mask > -1]
        y_window = mask[mask > -1]

        features = None

        for j in range(self.num_features):
            X_j = X_window[:, :, j]
            mean = X_j.mean(axis=1).reshape((-1, 1))
            features_j = [
                X_j.min(axis=1),
                X_j.max(axis=1),
                np.abs(X_j).min(axis=1),
                np.abs(X_j).max(axis=1),
                mean.flatten(),
                X_j.std(axis=1),
                (X_j > mean).sum(axis=1),
                (X_j <= mean).sum(axis=1)
            ]
            features_j = np.hstack([f.reshape((-1, 1)) for f in features_j])
            features = features_j if features is None else np.hstack((features, features_j))

        return X_window, features, y_window

    def port(self):
        """
        Port to C++
        :return: str
        """
        assert self.num_features is not None, 'unfitted'

        template = """
        #ifndef __WINDOW__{{ id }}
        #define __WINDOW__{{ id }}

        class Window {
            public:
                const uint16_t features_count = {{ features_count }};
                float features[{{ features_count }}];

                /**
                 * Extract features
                 */
                 bool transform(float *x, float *dest = NULL) {
                    // append source to queue
                    memcpy(queue + head, x, sizeof(float) * {{ num_features }});
                    head += {{ num_features }};

                    if (head != {{ size }}) {
                        return false;
                    }

                    // extract features for each axis
                    uint16_t feature_idx = 0;

                    for (uint16_t j = 0; j < {{ num_features }}; j++) {
                        float m = queue[j];
                        float M = m;
                        float abs_m = abs(m);
                        float abs_M = abs_m;
                        float mean = m;
                        float std = 0;
                        float count_above_mean = 0;
                        float count_below_mean = 0;

                        // first-order features
                        for (uint16_t i = j + {{ num_features }}; i < {{ size }}; i += {{ num_features }}) {
                            float xi = queue[i];
                            float abs_xi = abs(xi);

                            mean += xi;

                            if (xi < m) m = xi;
                            if (xi > M) M = xi;
                            if (abs_xi < abs_m) abs_m = abs_xi;
                            if (abs_xi > abs_M) abs_M = abs_xi;
                        }

                        mean /= {{ length }};

                        // second-order features
                        for (uint16_t i = j; i < {{ size }}; i += {{ num_features }}) {
                            float xi = queue[i];

                            std += (xi - mean) * (xi - mean);

                            if (xi > mean) count_above_mean += 1;
                            else count_below_mean += 1;
                        }

                        std = sqrt(std / {{ length }});

                        features[feature_idx++] = m;
                        features[feature_idx++] = M;
                        features[feature_idx++] = abs_m;
                        features[feature_idx++] = abs_M;
                        features[feature_idx++] = mean;
                        features[feature_idx++] = std;
                        features[feature_idx++] = count_above_mean;
                        features[feature_idx++] = count_below_mean;
                    }

                    // copy to dest, if any
                    if (dest != NULL) memcpy(dest, features, sizeof(float) * {{ features_count }});

                    // shift
                    memcpy(queue, queue + {{ shift }}, sizeof(float) * {{ overlap }});
                    head -= {{ shift }};

                    return true;
                 }

            protected:
                uint16_t head = 0;
                float queue[{{ size }}];
        };

        #endif
        """

        # interpolate variables
        variables = {
            'num_features': self.num_features,
            'features_count': 8 * self.num_features,
            'length': self.length,
            'size': self.length * self.num_features,
            'shift': self.shift * self.num_features,
            'overlap': (self.length - self.shift) * self.num_features,
            'id': id(self)
        }

        for k, v in variables.items():
            template = template.replace('{{ %s }}' % k, str(v))

        return template

    def _mode(self, yi):
        """
        Get mode of each sample if sufficient agreement is met
        :param y:
        :return:
        """
        unique, counts = np.unique(yi, return_counts=True)

        if max(counts) < self.agreement:
            return -1

        return unique[np.argmax(counts)]
