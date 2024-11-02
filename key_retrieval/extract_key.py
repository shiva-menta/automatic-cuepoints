from pyrekordbox import get_config

def main():
    print(f"Encryption Key is: {get_config("rekordbox6")["dp"]}")

if __name__ == "__main__":
    main()