import os
import sys


class Logger(object):
    def __init__(self, filename="unknown.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def log(main_func, log_filename="output.log"):
    original_stdout = sys.stdout
    sys.stdout = Logger(log_filename)

    try:
        main_func()
    except Exception as e:
        print(f"Experiment interrupted due to an exception: {e}")
    finally:
        sys.stdout.log.close()
        sys.stdout = original_stdout

        while True:
            save_log = input("Do you want to save the log file? (yes/no): ").lower()
            if save_log in ["yes", "y"]:
                break  # Keep the file and exit loop
            elif save_log in ["no", "n"]:
                os.remove(log_filename)  # Delete the file and exit loop
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
