class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate #Скорость обучения, шаг градиентного спуска
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.theta = None
        self.loss_history = [] #история потерь

    #Сигмоидальная функция линейная модель в вероятностную
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    #Добавляет столбец единиц для коэффициента смещения
    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X)) #объединение по столбцам

    def _initialize_parameters(self, n_features): #Инициализирует параметры модели небольшими случайными значениями.
        self.theta = np.random.normal(0, 0.01, n_features + 1)

    #Функция потерь- Измеряет, насколько предсказания отличаются от истинных значений
    #Формула: L = -1/n * Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
    def _compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    # Процесс обучения
    def fit(self, X, y):
        X_b = self._add_intercept(X)
        n_samples, n_features = X_b.shape

        self._initialize_parameters(X.shape[1])

        for iteration in range(self.n_iterations):

            linear_model = np.dot(X_b, self.theta) #матричное умножение
            #Вычисляем предсказания
            y_pred = self._sigmoid(linear_model)
            #Вычисляем градиент
            gradient = (1 / n_samples) * np.dot(X_b.T, (y_pred - y))
            #Обновляем параметры
            self.theta -= self.learning_rate * gradient
            #Сохраняем значение функции потерь
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

    # возвращает вероятности принадлежности классу 1
    def predict_proba(self, X):
        X_b = self._add_intercept(X)
        linear_model = np.dot(X_b, self.theta)
        return self._sigmoid(linear_model)

    # возвращает бинарные предсказания (0 или 1) на основе порога
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    #вычисляет точность предсказаний
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def plot_loss_history(self):
        """Визуализация истории потерь"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.title('График функции потерь')
        plt.xlabel('Итерация')
        plt.ylabel('Потери')
        plt.grid(True)
        plt.show()

    def get_params(self):
        return {
            'theta': self.theta,
            'bias': self.theta[0],
            'weights': self.theta[1:],
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations
        }

    def get_decision_boundary(self, X, threshold=0.5):
        if X.shape[1] != 2:
            raise ValueError("Метод работает только для 2 признаков")

        boundary_constant = np.log(threshold / (1 - threshold))

        def boundary_function(x1):
            return (-self.theta[0] - self.theta[1] * x1 + boundary_constant) / self.theta[2]

        return boundary_function
