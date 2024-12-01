#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <numeric>

void process_data_sequential_implicit(const std::string& input_file, const std::string& output_file){
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);

    if (!infile.is_open()) {
        std::cerr << "Error in opening output file: " << input_file << std::endl;
        return;
    }
    if (!outfile.is_open()) {
        std::cerr << "Error in opening input file: " << output_file << std::endl;
        return;
    }

    std::vector<double> numbers;
    double value;
    
    
    while (infile >> value) {
        numbers.push_back(value);
    }

    
    std::vector<int> n_values = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int block_size = 200;

    
    for (size_t i = 0; i < n_values.size(); ++i) {
        int n = n_values[i];
        size_t start_index = i * block_size;  
        size_t end_index = start_index + block_size;  

        if (end_index > numbers.size()) {
            std::cerr << "Insufficient data for block " << n << " x " << n << std::endl;
            break;
        }

        double sum_odd = 0.0, sum_even = 0.0;
        int odd_count = 0, even_count = 0;

        for (size_t j = start_index; j < end_index; ++j) {
            if ((j - start_index) % 2 == 0) { 
                sum_odd += numbers[j];
                odd_count++;
            } else { 
                sum_even += numbers[j];
                even_count++;
            }
        }

        
        double avg_odd = (odd_count > 0) ? sum_odd / odd_count : 0.0;
        double avg_even = (even_count > 0) ? sum_even / even_count : 0.0;

        
        outfile << n << " x " << n << ":\n";
        outfile << std::fixed << std::setprecision(9) << avg_odd << "\n";
        outfile << std::fixed << std::setprecision(9) << avg_even << "\n";
    }

    infile.close();
    outfile.close();
}

void process_data_explicit(const std::string& input_file, const std::string& output_file){
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);

    if (!infile.is_open()) {
        std::cerr << "Error opening input file: " << input_file << std::endl;
        return;
    }
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_file << std::endl;
        return;
    }

    std::vector<double> numbers;
    double value;

    while (infile >> value) {
        numbers.push_back(value);
    }

    std::vector<int> n_values = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> x_values = {1, 2, 4, 8, 16, 20};
    int block_size = 200;
    size_t current_line = 0;
    int x_index = 0;

    size_t total_lines = numbers.size();
    
    while (current_line < total_lines) {
        for (size_t i = 0; i < n_values.size(); ++i) {
            int n = n_values[i];
            size_t start_index = current_line;
            size_t end_index = start_index + block_size;

            if (end_index > total_lines) {
                std::cerr << "Not enough data for block " << n << " x " << n << std::endl;
                break;
            }

            int x = x_values[x_index];

            double sum_odd = 0.0, sum_even = 0.0;
            int odd_count = 0, even_count = 0;

            for (size_t j = start_index; j < end_index; ++j) {
                if ((j - start_index) % 2 == 0) {
                    sum_odd += numbers[j];
                    odd_count++;
                } else {
                    sum_even += numbers[j];
                    even_count++;
                }
            }

            double avg_odd = (odd_count > 0) ? sum_odd / odd_count : 0.0;
            double avg_even = (even_count > 0) ? sum_even / even_count : 0.0;

            outfile << "number of threads: " << x << "\n";
            outfile << n << " x " << n << ":\n";
            outfile << std::fixed << std::setprecision(9) << avg_odd << "\n";
            outfile << std::fixed << std::setprecision(9) << avg_even << "\n";

            current_line = end_index;

            if (n == 4096) {
                x_index = (x_index + 1) % x_values.size();
            }
        }
    }

    infile.close();
    outfile.close();
}

void process_data_improvement(const std::string& inputFile, const std::string& outputFile){
    std::ifstream inFile(inputFile);
    std::ofstream outFile(outputFile);

    if (!inFile) {
        std::cerr << "Error: Unable to open input file!" << std::endl;
        return;
    }

    if (!outFile) {
        std::cerr << "Error: Unable to open output file!" << std::endl;
        return;
    }

    
    std::vector<int> threads = {1, 2, 4, 8, 16, 20};
    std::vector<int> matrixSizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int linesPerThread = 1800;
    const int linesPerMatrix = 100;
    const int blockSize = 200; 

    std::vector<double> data;
    double value;

    while (inFile >> value) {
        data.push_back(value);
    }

    int totalLines = data.size();
    if (totalLines != 10800) {
        std::cerr << "Error: Expected 10800 lines in the input file, but got " << totalLines << "." << std::endl;
        return;
    }

    int threadIndex = 0;
    for (int start = 0; start < totalLines; start += linesPerThread) {
        int x = threads[threadIndex++ % threads.size()];
        outFile << "threads number: " << x << "\n";

        int matrixIndex = 0;
        for (int i = 0; i < linesPerThread; i += linesPerMatrix) {
            int n = matrixSizes[matrixIndex++ % matrixSizes.size()];
            outFile << "n x n: " << n << "\n";

            double sumOdd = 0.0, sumEven = 0.0;
            int countOdd = 0, countEven = 0;

            for (int j = 0; j < blockSize; ++j) {
                int lineIndex = start + i + j;
                if (lineIndex >= totalLines) break;

                if (j % 2 == 0) {
                    sumEven += data[lineIndex];
                    countEven++;
                } else {
                    sumOdd += data[lineIndex];
                    countOdd++;
                }
            }

            double avg_1 = (countOdd > 0) ? (sumOdd / countOdd) : 0.0;
            double avg_2 = (countEven > 0) ? (sumEven / countEven) : 0.0;

            outFile << std::fixed << std::setprecision(6) << avg_1 << "\n";
            outFile << std::fixed << std::setprecision(6) << avg_2 << "\n";
        }
    }

    inFile.close();
    outFile.close();
}

int main(int argc, char* argv[]){
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <process_option>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    int process_option = atoi(argv[2]);

    std::string output_file;


    switch (process_option) {
        case 1:
            output_file = "ProcessedData_Sequential";
            process_data_sequential_implicit(input_file, output_file);
            std::cout << "Elaboration complete, saved data in " << output_file << std::endl;
            break;
        case 2:
            output_file = "ProcessedData_Implicit";
            process_data_sequential_implicit(input_file, output_file);
            std::cout << "Elaboration complete, saved data in " << output_file << std::endl;
            break;
        case 3:
            output_file = "ProcessedData_Explicit";
            process_data_explicit(input_file, output_file);
            std::cout << "Elaboration complete, saved data in " << output_file << std::endl;
            break;
        case 4:
            output_file = "ProcessedData_Improvement";
            process_data_improvement(input_file, output_file);
            std::cout << "Elaboration complete, saved data in " << output_file << std::endl;
            break;
    }

    return 0;
}