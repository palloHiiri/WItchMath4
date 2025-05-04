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

# === Дополнительные метрики ===
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

# === Чтение данных ===
def load_data():
    print("Выберите способ ввода данных:")
    print("1. Ввести вручную")
    print("2. Загрузить из файла")

    choice = input("Ваш выбор: ").strip()
    if choice == "1":
        n = int(input("Введите количество точек (от 8 до 12): "))
        if not (8 <= n <= 12):
            raise ValueError("Количество точек должно быть от 8 до 12.")
        x = list(map(float, input(f"Введите {n} значений x через пробел: ").split()))
        y = list(map(float, input(f"Введите {n} значений y через пробел: ").split()))
        return np.array(x), np.array(y)
    elif choice == "2":
        path = input("Введите путь к файлу (формат: x1 y1\nx2 y2...): ")
        if not os.path.exists(path):
            raise FileNotFoundError("Файл не найден.")
        data = np.loadtxt(path)
        x, y = data[:, 0], data[:, 1]
        if not (8 <= len(x) <= 12):
            raise ValueError("Неверное количество точек в файле.")
        return x, y
    else:
        raise ValueError("Неверный выбор.")

# === Форматированный вывод ===
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

# === График ===
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

# === Точка входа ===
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
        print("\n🏆 Лучшая аппроксимирующая функция:", best[0])
        print("=" * 50)

        # Отображаем график
        plot_all_functions(x, y, [(r[0], r[1], r[2]) for r in results])

    except Exception as e:
        print(f"[Ошибка] {str(e)}")

if __name__ == "__main__":
    main()