from pyrekordbox import get_config

def main():
    print(f"Encryption Key is: {get_config("rekordbox6")["dp"]}")
    print(f"Master DB path is: {get_config("rekordbox6")["db_path"]}")

if __name__ == "__main__":
    main()