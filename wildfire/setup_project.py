from pathlib import Path
import click

from cookiecutter.main import cookiecutter
from cookiecutter.exceptions import RepositoryNotFound


if __name__ == '__main__':
    print(__file__ + '/template')
    try:
        repository_path = cookiecutter(
            str(Path(__file__).parent / 'template')
        )
    except RepositoryNotFound as exception:
        print('__file__:', __file__)
        raise exception
    
    repository_path = Path(repository_path)

    move_to_current = click.prompt(
        'Do you want to setup the project in the current directory? '
        f'Otherwise a new directory {repository_path.name} is created',
        default='y'
    )

    if move_to_current == 'y':
        temporary_path = repository_path.parent / 'temporary'
        repository_path.replace(temporary_path)

        for path in Path(temporary_path).glob('*'):
            path.replace(temporary_path.parent / path.name)

        temporary_path.rmdir()
