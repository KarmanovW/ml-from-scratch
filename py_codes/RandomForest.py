class DecisionTree:
    class _Node:
        def __init__(self, index, t, true_branch, false_branch):
            self.index = index
            self.t = t
            self.true_branch = true_branch
            self.false_branch = false_branch

    class _Leaf:
        def __init__(self, labels):
            self.prediction = self.predict(labels)

        def predict(self, labels):
            classes, counts = np.unique(labels, return_counts=True)
            return classes[np.argmax(counts)]

    def __init__(self, min_leaf=5, max_depth=None, criterion='gini'):
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None

    def _gini(self, labels):
        if len(labels) == 0:
            return 0

        counts = np.unique(labels, return_counts=True)[1]
        p = counts / len(labels)

        return 1 - np.sum(p ** 2)

    def _entropy(self, labels):
        if len(labels) == 0:
            return 0

        counts = np.unique(labels, return_counts=True)[1]
        p = counts / len(labels)

        return -np.sum(p * np.log2(p + 1e-12))

    def _impurity(self, labels):
        if self.criterion == "entropy":
            return self._entropy(labels)

        return self._gini(labels)

    def _quality(self, left_labels, right_labels, current_impurity):
        S = len(left_labels) / (len(left_labels) + len(right_labels))
        return current_impurity - S * self._impurity(left_labels) - (1 - S) * self._impurity(right_labels)

    def _split(self, data, labels, index, t):
        true_mask = data[:, index] <= t
        false_mask = data[:, index] > t

        return data[true_mask], data[false_mask], labels[true_mask], labels[false_mask]

    def _find_best_split(self, data, labels):
        best_quality = 0
        best_t = None
        best_index = None

        current_impurity = self._impurity(labels)
        n_features = data.shape[1]

        for index in range(n_features):
            values = np.unique(data[:, index])

            if len(values) == 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for t in thresholds:
                true_labels, false_labels = self._split(data, labels, index, t)[2:]

                if len(true_labels) < self.min_leaf or len(false_labels) < self.min_leaf:
                    continue

                q = self._quality(true_labels, false_labels, current_impurity)

                if q > best_quality:
                    best_quality = q
                    best_t = t
                    best_index = index

        return best_quality, best_t, best_index

    def _build_tree(self, data, labels, depth=0):
        if self.max_depth is not None and depth >= self.max_depth:
            return self._Leaf(labels)

        best_quality, best_t, best_index = self._find_best_split(data, labels)

        if best_quality == 0:
            return self._Leaf(labels)

        true_data, false_data, true_labels, false_labels = self._split(
            data, labels, best_index, best_t
        )

        true_branch = self._build_tree(true_data, true_labels, depth + 1)
        false_branch = self._build_tree(false_data, false_labels, depth + 1)

        return self._Node(best_index, best_t, true_branch, false_branch)

    def fit(self, X, y):
        self.tree = self._build_tree(np.array(X), np.array(y))

    def _classify_object(self, obj, node):
        if isinstance(node, self._Leaf):
            return node.prediction

        if obj[node.index] <= node.t:
            return self._classify_object(obj, node.true_branch)

        return self._classify_object(obj, node.false_branch)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._classify_object(obj, self.tree) for obj in X])

    def score(self, X, y):  # вычисляет долю верных предсказаний (accuracy)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class RandomForest:
    def __init__(self, n_trees=10, min_leaf=5, max_depth=None, criterion='gini'):
        self.n_trees = n_trees          # количество деревьев в лесу
        self.min_leaf = min_leaf        # минимальный размер листа для каждого дерева
        self.max_depth = max_depth      # максимальная глубина дерева
        self.criterion = criterion      # критерий разбиения ('gini' или 'entropy')
        self.forest = []                # список для хранения деревьев и их подвыборок признаков


    def _get_bootstrap(self, data, labels):
        # Объекты в бутстрэп выборке — строки исходного массива data, выбранные случайно с повторением.
        n_samples = data.shape[0]                      # число объектов в данных
        indices = np.random.randint(0, n_samples, size=n_samples)  # случайные индексы с повторениями
        return data[indices], labels[indices]          # возвращаем подвыборку объектов и их меток


    def _get_subsample(self, n_features):
        # Создаёт случайную подвыборку признаков для каждого дерева
        feature_indexes = list(range(n_features))     # список индексов всех признаков
        np.random.shuffle(feature_indexes)            # перемешиваем индексы
        len_subsample = int(np.sqrt(n_features))      # размер подвыборки = sqrt(число признаков)
        return feature_indexes[:len_subsample]        # возвращаем подвыборку индексов


    def fit(self, data, labels):
        # Обучение Random Forest на данных
        n_features = data.shape[1]                    # число признаков
        self.forest = []                              # очищаем лес перед обучением

        for _ in range(self.n_trees):                 # обучаем каждое дерево
            b_data, b_labels = self._get_bootstrap(data, labels)  # создаём бутстрэп выборку объектов
            subsample = self._get_subsample(n_features)           # создаём подвыборку признаков

            # Создаём дерево и обучаем его на бутстрэп выборке с подвыборкой признаков
            tree = DecisionTree(min_leaf=self.min_leaf, max_depth=self.max_depth, criterion=self.criterion)
            tree.fit(b_data[:, subsample], b_labels)

            # Сохраняем дерево и индексы признаков, на которых оно обучалось
            self.forest.append((tree, subsample))


    def _tree_vote(self, data):
        # Голосование всех деревьев для предсказания классов
        n_samples = data.shape[0]                     # число объектов для предсказания
        n_trees = len(self.forest)                    # число деревьев в лесу
        all_preds = np.empty((n_trees, n_samples), dtype=object)  # массив для хранения предсказаний каждого дерева

        # Предсказания каждого дерева
        for i, (tree, subsample) in enumerate(self.forest):
            all_preds[i] = tree.predict(data[:, subsample])       # предсказания дерева для выбранных признаков

        voted_predictions = []                          # список для финальных предсказаний леса

        # Голосование для каждого объекта
        for j in range(n_samples):
            counts = {}                                 # словарь для подсчёта голосов классов

            for i in range(n_trees):
                cls = all_preds[i, j]                  # класс, предсказанный i-м деревом для j-го объекта
                counts[cls] = counts.get(cls, 0) + 1  # считаем количество голосов за каждый класс

            # Определяем класс с наибольшим количеством голосов
            max_count = -1
            chosen_class = None

            for cls, cnt in counts.items():
                if cnt > max_count:
                    max_count = cnt
                    chosen_class = cls

            voted_predictions.append(chosen_class)     # добавляем выбранный класс в список

        return np.array(voted_predictions)             # возвращаем массив финальных предсказаний


    def predict(self, data):
        # Предсказание классов для новых объектов
        return self._tree_vote(data)                   # вызываем метод голосования деревьев


    def score(self, data, labels):
        # Вычисляет точность (accuracy) модели на данных
        y_pred = self.predict(data)                    # получаем предсказания
        return np.mean(y_pred == labels)              # среднее число верных предсказаний
