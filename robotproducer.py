import sys

def main():
    # is input text optional?
    if len(sys.argv) != 2: 
        print("Usage: robotproducer.py [input.txt]")
        with open(sys.argv[1], "r") as f:
            text = f.read()
    

if __name__ == "__main__":
    main()