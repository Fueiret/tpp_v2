#!/bin/bash

# Имя исходного файла и бинарника
SRC_FILE="matrix_multiplication_2.cpp"
EXEC_FILE="matrix_mult"

# Проверка передачи параметра L
if [ -z "$1" ]; then
    echo "Использование: $0 <размер L>"
    echo "Пример: $0 500"
    exit 1
fi

L_SIZE=$1

echo "=== Компиляция программы ==="
mpicc -o "$EXEC_FILE" "$SRC_FILE" -lm

if [ $? -ne 0 ]; then
    echo "Ошибка компиляции! Проверьте код."
    exit 1
fi

echo ""
echo "=== Запуск экспериментов (L = $L_SIZE, N = $((L_SIZE * 10))) ==="
echo "----------------------------------------------------------------------------------"
printf "%-12s | %-12s | %-12s | %-14s | %-20s\n" "Процессы (p)" "Время (с)" "Ускорение S*" "Эффективность E*" "Норма ||C||^2"
echo "----------------------------------------------------------------------------------"

# Массив значений числа процессов (p) согласно методичке УГАТУ
PROCESS_COUNTS=(1 2 4 8 16 32)

# Переменная для хранения времени на 1 процессе T_1
T1=0

for p in "${PROCESS_COUNTS[@]}"; do
    # Запуск mpirun и сбор вывода
    output=$(mpirun -np "$p" ./"$EXEC_FILE" "$L_SIZE")

    # Парсинг времени и нормы из вывода программы
    elapsed_time=$(echo "$output" | grep "Elapszed time" | awk '{print $3}')
    norm_val=$(echo "$output" | grep "|C|^2" | awk '{print $3}')

    # Если время не распарсилось, пропускаем расчет
    if [ -z "$elapsed_time" ]; then
        echo "Ошибка выполнения при p = $p"
        continue
    fi

    # Фиксируем T_1 при первом запуске (p = 1)
    if [ "$p" -eq 1 ]; then
        T1=$elapsed_time
    fi

    # Вычисление ускорения (S_p = T_1 / T_p) и эффективности (E_p = S_p / p) через bc
    speedup=$(echo "scale=4; $T1 / $elapsed_time" | bc -l)
    efficiency=$(echo "scale=4; $speedup / $p" | bc -l)

    # Вывод строки таблицы
    printf "%-12d | %-12.6f | %-12.4f | %-14.4f | %-20s\n" "$p" "$elapsed_time" "$speedup" "$efficiency" "$norm_val"
done

echo "----------------------------------------------------------------------------------"
echo "Эксперименты завершены!"
