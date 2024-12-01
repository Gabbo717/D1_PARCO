import matplotlib.pyplot as plt
from collections import defaultdict

def read_and_plot_sequential(file_path):
    matrix_sizes = []
    checkSymSequential_times = []
    matTransposeSequential_times = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        matrix_size = int(lines[i].split()[0])
        matrix_sizes.append(matrix_size)
        
        checkSymSequential_times.append(float(lines[i + 1].strip()))
        matTransposeSequential_times.append(float(lines[i + 2].strip()))

    plt.figure(figsize=(10, 5))
    plt.plot(matrix_sizes, checkSymSequential_times, color='blue', marker='o', label='checkSymSequential')
    plt.title('checkSymSequential - matrix size vs elaboration time')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Elaboration Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(matrix_sizes)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(matrix_sizes, matTransposeSequential_times, color='red', marker='o', label='matTransposeSequential')
    plt.title('matTransposeSequential - matrix size vs elaboration time')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Elaboration Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(matrix_sizes)
    plt.show()

def read_and_plot_implicit_par(file_path):
    matrix_sizes = []
    checkSymSequential_times = []
    matTransposeSequential_times = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        matrix_size = int(lines[i].split()[0])
        matrix_sizes.append(matrix_size)
        
        checkSymSequential_times.append(float(lines[i + 1].strip()))
        matTransposeSequential_times.append(float(lines[i + 2].strip()))

    plt.figure(figsize=(10, 5))
    plt.plot(matrix_sizes, checkSymSequential_times, color='blue', marker='o', label='checkSymImplicit')
    plt.title('checkSymImplicitPar - matrix size vs elaboration time')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Elaboration Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(matrix_sizes)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(matrix_sizes, matTransposeSequential_times, color='red', marker='o', label='matTransposeImplicit')
    plt.title('matTransposeImplicitPar - matrix size vs elaboration time')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Elaboration Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(matrix_sizes)
    plt.show()


def read_and_plot_explicit_par(file_path):
    data = {}
    current_threads = None

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("number of threads:"):
            current_threads = int(line.split(":")[1].strip())
            if current_threads not in data:
                data[current_threads] = {'matrix_sizes': [], 'checkSym': [], 'matTranspose': []}
        elif "x" in line:
            matrix_size = int(line.split("x")[0].strip())
            data[current_threads]['matrix_sizes'].append(matrix_size)
        else:
            value = float(line)
            if len(data[current_threads]['checkSym']) < len(data[current_threads]['matrix_sizes']):
                data[current_threads]['checkSym'].append(value)
            else:
                data[current_threads]['matTranspose'].append(value)

    plt.figure(figsize=(10, 5))
    for threads, values in data.items():
        plt.plot(values['matrix_sizes'], values['checkSym'], marker='o', label=f"threads: {threads}")
    plt.title('checkSymExplicitPar - matrix size vs elaboration time')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Elaboration Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(values['matrix_sizes'])
    plt.show()

    plt.figure(figsize=(10, 5))
    for threads, values in data.items():
        plt.plot(values['matrix_sizes'], values['matTranspose'], marker='o', label=f"threads: {threads}")
    plt.title('matTransposeExplicitPar - matrix size vs elaboration time')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Elaboration Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(values['matrix_sizes'])
    plt.show()

def parse_and_plot_improvement_percentage(file_path):
    thread_data = defaultdict(lambda: defaultdict(list))  

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_thread = None
    current_n = None
    value_count = 0  

    for line in lines:
        line = line.strip()
        if line.startswith("threads number:"):
            current_thread = int(line.split(":")[1].strip())
        elif line.startswith("n x n:"):
            current_n = int(line.split(":")[1].strip())
            value_count = 0  
        else:
            
            try:
                value = float(line)
                value_count += 1
                if value_count == 2:
                    
                    thread_data[current_thread][current_n].append(value)
                elif value_count > 2:
                    
                    raise ValueError(f"Unexpected extra values for matrix size {current_n} in thread {current_thread}")
            except ValueError:
                print(f"Skipping invalid line: {line}")

    plot_data_improvement_percentage(thread_data)

def plot_data_improvement_percentage(thread_data):
    plt.figure(figsize=(12, 6))

    
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    for i, (thread, data) in enumerate(thread_data.items()):
        if thread == 1: 
            continue
        x_thread = sorted(data.keys())
        y_thread = [sum(data[n]) / len(data[n]) for n in x_thread]
        plt.plot(x_thread, y_thread, label=f"thread: {thread}", color=colors[i % len(colors)], marker="o")

    
    plt.xlabel("Matrix Sizes")
    plt.ylabel("Improvement (%)")
    plt.title("Performance Graph")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xscale('log')
    plt.yscale('log')

    plt.show()

if __name__ == "__main__":
    file_path = input("Enter the file path for the data: ")

    decisions_path = {
        '1': read_and_plot_sequential,
        '2': read_and_plot_implicit_par,
        '3': read_and_plot_explicit_par,
        '4': parse_and_plot_improvement_percentage
    }

    print("Plotting options:\n1 - Sequential (baseline)\n2 - Implicit Parallelization\n3 - Explicit Parallelization\n4 - Speedup Improvement")
    plot_option = input("Enter plotting option: ")
    if plot_option in decisions_path:
        decisions_path[plot_option](file_path)
    else:
        print("wrong input!")
        