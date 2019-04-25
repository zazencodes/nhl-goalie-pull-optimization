import sys
import os
import glob
import click

@click.command()
@click.option(
    '--whitelist',
    '-w',
    default='',
    help='File names to ignore (keep). Comma separated.'
)
@click.option(
    '--ignore-archive',
    is_flag=True,
    help='Ignore files in archive directory'
)
def main(whitelist, ignore_archive):
    print('''
        Cleaning notebook files from archive, py and html folders. 
        Only keeping files that exist in src.
    ''')
    dump = []
    strip_filename = lambda f: os.path.splitext(
        os.path.split(f)[-1]
    )[0]
    strip_archive_filename = lambda f: os.path.splitext(
        os.path.split(f)[-1]
    )[0][:-11] # must strip off the date

    src_files = glob.glob('src/*.ipynb')
    src_files_stripped = [strip_filename(f) for f in src_files]
    if whitelist:
        print('Adding whitelist file(s) to source...')
        src_files_stripped += [w.strip() for w in whitelist.split(',')]
    print('Source files:')
    print('\n'.join(src_files_stripped))
    print()

    if not ignore_archive:
        print('Searching notebooks/archive ...')
        files = glob.glob(os.path.join('archive', '*.ipynb'))
        files_stripped = [strip_archive_filename(f) for f in files]
        for f, path in zip(files_stripped, files):
            if not any((f==src_f for src_f in src_files_stripped)):
                dump.append(path)

    print('Searching notebooks/py ...')
    files = glob.glob(os.path.join('py', '*.py'))
    files_stripped = [strip_filename(f) for f in files]
    for f, path in zip(files_stripped, files):
        if not any((f==src_f for src_f in src_files_stripped)):
            dump.append(path)

    print('Searching notebooks/html ...')
    files = glob.glob(os.path.join('html', '*.html'))
    files_stripped = [strip_filename(f) for f in files]
    for f, path in zip(files_stripped, files):
        if not any((f==src_f for src_f in src_files_stripped)):
            dump.append(path)

    print()
    q = 'Found {} files to delete:\n'.format(len(dump))
    q += '\n'.join(dump) + '\n\n'
    q += 'Proceed? (y/n)'
    ans = input(q)
    if ans != 'y':
        sys.exit('Not removing files. Exiting.')
    for f in dump:
        print('rm {}'.format(f))
        os.remove(f)

if __name__ == '__main__':
    main()
