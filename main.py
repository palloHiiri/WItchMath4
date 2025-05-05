import numpy as np
import matplotlib.pyplot as plt
import os

def linear_approximation(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    phi = lambda x_: a * x_ + b
    return phi, (a, b)

def quadratic_approximation(x, y):
    A = np.vstack([x**2, x, np.ones(len(x))]).T
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    phi = lambda x_: a * x_**2 + b * x_ + c
    return phi, (a, b, c)

def cubic_approximation(x, y):
    A = np.vstack([x**3, x**2, x, np.ones(len(x))]).T
    a, b, c, d = np.linalg.lstsq(A, y, rcond=None)[0]
    phi = lambda x_: a * x_**3 + b * x_**2 + c * x_ + d
    return phi, (a, b, c, d)

def exponential_approximation(x, y):
    if np.any(y <= 0):
        raise ValueError("y должен быть > 0 для экспоненциальной аппроксимации.")
    log_y = np.log(y)
    A = np.vstack([x, np.ones(len(x))]).T
    a, ln_b = np.linalg.lstsq(A, log_y, rcond=None)[0]
    b = np.exp(ln_b)
    phi = lambda x_: b * np.exp(a * x_)
    return phi, (a, b)

def logarithmic_approximation(x, y):
    if np.any(x <= 0):
        raise ValueError("x должен быть > 0 для логарифмической аппроксимации.")
    log_x = np.log(x)
    A = np.vstack([log_x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    phi = lambda x_: a * np.log(np.clip(x_, 1e-10, None)) + b
    return phi, (a, b)

def power_approximation(x, y):
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("x и y должны быть > 0 для степенной аппроксимации.")
    log_x = np.log(x)
    log_y = np.log(y)
    b, ln_a = np.linalg.lstsq(np.vstack([log_x, np.ones(len(x))]).T, log_y, rcond=None)[0]
    a = np.exp(ln_a)
    phi = lambda x_: a * (np.clip(x_, 1e-10, None)**b)
    return phi, (a, b)

def calculate_sse(phi, x, y):
    return sum((phi(xi) - yi) ** 2 for xi, yi in zip(x, y))

def calculate_sigma(phi, x, y):
    return np.sqrt(calculate_sse(phi, x, y) / len(x))

def correlation_coefficient(x, y):
    return np.corrcoef(x, y)[0, 1]

def coefficient_of_determination(phi, x, y):
    y_mean = np.mean(y)
    sst = sum((yi - y_mean) ** 2 for yi in y)
    sse = calculate_sse(phi, x, y)
    return 1 - sse / sst if sst != 0 else 0


def load_data():
    while True:
        print("\nВыберите способ ввода данных:")
        print("1. Ввести вручную")
        print("2. Загрузить из файла")
        print("3. Выход")

        choice = input("Ваш выбор: ").strip()

        if choice == "3":
            print("Выход")
            exit()

        try:
            if choice == "1":
                while True:
                    n_input = input("Введите количество точек (от 8 до 12): ")
                    if not n_input.isdigit():
                        print("А можно пж число, а можно пж целое?")
                        continue
                    n = int(n_input)
                    if 8 <= n <= 12:
                        break
                    print("Количество точек должно быть от 8 до 12. Другие значения не принимаем")

                while True:
                    x_input = input(f"Введите {n} значений x через пробел: ").replace(',', '.')
                    if not all(c in '0123456789.- ' for c in x_input):
                        print("Моя регулярка говорит, что ты ввел плохо, я ей верю.")
                        continue
                    x = list(map(float, x_input.split()))
                    if len(x) == n:
                        break
                    print(f"Ожидалось {n} значений, получено {len(x)}. Что-то тут не так...")

                while True:
                    y_input = input(f"Введите {n} значений y через пробел: ").replace(',', '.')
                    if not all(c in '0123456789.- ' for c in y_input):
                        print("Моя регулярка говорит, что ты правила нарушаешь, давай заново")
                        continue
                    y = list(map(float, y_input.split()))
                    if len(y) == n:
                        break
                    print(f"Ожидалось {n} значений, получено {len(y)}. Мы тут циферки прикинули... не сходятся!")

                points = set(zip(x, y))
                if len(points) < n:
                    print("А че эт у нас точки повторяются? Нееееееее, низя!")
                    continue

                return np.array(x), np.array(y)

            elif choice == "2":
                path = input("Введите путь к файлу (формат файла: x1 y1\nx2 y2...): ")

                try:
                    if not os.path.exists(path):
                        print("Такого файла у нас нет")
                        continue

                    with open(path, 'r') as f:
                        content = f.read().replace(',', '.')

                    if not all(c in '0123456789. - \n' for c in content):
                        print("Моя регулярка говорит, что в файле что-то не то")
                        continue

                    data = np.loadtxt(content.splitlines())

                    if data.shape[1] != 2 or not (8 <= len(data) <= 12):
                        print("В файле должно быть от 8 до 12 пар точек, не больше, не меньше, только так")
                        continue

                    points = set(tuple(point) for point in data)
                    if len(points) < len(data):
                        print("Точки повторяться не могут, не надо тут!")
                        continue

                    return data[:, 0], data[:, 1]

                except ValueError as e:
                    print("Ошибка при чтении файла, что-то ты там накосячил")
                    continue
                except Exception as e:
                    print("Ну такого даже я не ожидала")
                    continue

            else:
                print("Неверный выбор. Пожалуйста, введите 1, 2 или 0.")

        except KeyboardInterrupt:
            print("\nКлавиши жмете всякие... я умер, все!")
            exit()

def format_function(name, coeffs, sigma, r2, sse, corr=None):
    print("=" * 50)
    print(f"Аппроксимирующая функция: {name}")
    if name == "Линейная":
        print(f"Функция: φ(x) = {coeffs[0]:.3f}x + {coeffs[1]:.3f}")
        print(f"Коэффициент корреляции Пирсона: r = {corr:.3f}")
    elif name == "Полиноминальная 2-й степени":
        print(f"Функция: φ(x) = {coeffs[0]:.3f}x² + {coeffs[1]:.3f}x + {coeffs[2]:.3f}")
    elif name == "Полиноминальная 3-й степени":
        print(f"Функция: φ(x) = {coeffs[0]:.3f}x³ + {coeffs[1]:.3f}x² + {coeffs[2]:.3f}x + {coeffs[3]:.3f}")
    elif name == "Экспоненциальная":
        print(f"Функция: φ(x) = {coeffs[1]:.3f} * e^({coeffs[0]:.3f}x)")
    elif name == "Логарифмическая":
        print(f"Функция: φ(x) = {coeffs[0]:.3f} * ln(x) + {coeffs[1]:.3f}")
    elif name == "Степенная":
        print(f"Функция: φ(x) = {coeffs[0]:.3f} * x^{coeffs[1]:.3f}")
    print(f"Среднеквадратичное отклонение: σ = {sigma:.3f}")
    print(f"Коэффициент детерминации: R² = {r2:.3f}")
    print(f"Мера отклонения: S = {sse:.3f}")
    print("=" * 50)

def plot_all_functions(x, y, functions):
    x_plot = np.linspace(min(x) - 0.5, max(x) + 0.5, 500)
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, color='black', label='Исходные точки')
    for name, phi, _ in functions:
        plt.plot(x_plot, [phi(xi) for xi in x_plot], label=name)
    plt.grid(True)
    plt.legend()
    plt.title('Графики аппроксимаций')
    plt.xlabel('x')
    plt.ylabel('φ(x)')
    plt.show()

def main():
    try:
        x, y = load_data()
        models = [
            ("Линейная", linear_approximation),
            ("Полиноминальная 2-й степени", quadratic_approximation),
            ("Полиноминальная 3-й степени", cubic_approximation),
            ("Экспоненциальная", exponential_approximation),
            ("Логарифмическая", logarithmic_approximation),
            ("Степенная", power_approximation),
        ]
        results = []

        for name, func in models:
            try:
                phi, coeffs = func(x, y)
                sigma = calculate_sigma(phi, x, y)
                r2 = coefficient_of_determination(phi, x, y)
                sse = calculate_sse(phi, x, y)
                corr = correlation_coefficient(x, y) if name == "Линейная" else None
                results.append((name, phi, coeffs, sigma, r2, sse, corr))
                format_function(name, coeffs, sigma, r2, sse, corr)
            except Exception as e:
                print(f"[Ошибка] Не удалось выполнить {name}: {str(e)}")

        best = min(results, key=lambda r: r[3])
        print("\nИ победа в номинации лучшая аппроксимирующая функция достается: :", best[0])
        print("=" * 50)

        # Отображаем график
        plot_all_functions(x, y, [(r[0], r[1], r[2]) for r in results])

    except Exception as e:
        print(f"Сломалось:( {str(e)}")

if __name__ == "__main__":
    main()