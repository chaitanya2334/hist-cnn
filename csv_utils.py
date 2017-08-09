import csv


def read(filename):
    with open(filename) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, dialect=dialect)
        rows = list(reader)
        return rows
