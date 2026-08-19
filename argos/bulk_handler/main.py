from questdb.ingress import Sender

from config import QUESTDB_CONF

from importer import import_file
from pathlib import Path


#
# FILES = [
#
#     # Add your files here# "path/to/file.dat"
# ]
directory = "/data4bk/nirb/Development/guyc/יולי 3"     #"path/to/directory"
FILES  = Path(directory).glob("*.dat")

def main():

    with Sender.from_conf(
        QUESTDB_CONF
    ) as sender:


        for file in FILES:

            try:

                import_file(
                    sender,
                    file,
                )


            except Exception as e:

                print()
                print(
                    "FAILED:"
                )

                print(file)

                print(e)



if __name__ == "__main__":

    main()