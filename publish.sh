#!/usr/bin/env bash
# ./publish.sh 0.13.37
set -o errexit -o nounset -o pipefail -o xtrace

VERSION="$1"
VERSION_TAG="v${VERSION}"

publish () {
    (
        VERSION="$1"

        python setup.py sdist

        read -r -p "Looks good to publish ${VERSION} to pypi? " response
        case "$response" in
            [yY][eE][sS]|[yY])
                twine upload dist/ml-workflow-${VERSION}.tar.gz
                ;;
            *)
                echo "Aborted upload"
                exit 5
                ;;
        esac
    )
}

(
    cd "$( dirname "${BASH_SOURCE[0]}" )"

    git fetch
    test -z "$(git status --porcelain)" || (echo "Dirty repo"; exit 2)
    test -z "$(git diff origin/master)" || (echo "Not up to date with origin/master"; exit 3)


    git fetch --tags
    git tag -l | sed '/^'"${VERSION_TAG}"'$/{q2}' > /dev/null \
        || (echo "${VERSION_TAG} already exists"; exit 4)

    sed -i -E 's/^(\s*)version(\s*)=(\s*)\x27.*\x27,$/\1version\2=\3\x27'"${VERSION}"'\x27,/g' setup.py

    pytest

    git diff origin/master

    read -r -p "Deploying ${VERSION_TAG}, are you sure? [y/N]? " response
    case "$response" in
        [yY][eE][sS]|[yY])
            git add setup.py
            git commit -m"Version ${VERSION_TAG}"
            git tag "${VERSION_TAG}"
            git push origin "${VERSION_TAG}"
            git push origin master

            publish ${VERSION}
            ;;
        *)
            git checkout .
            echo "Aborted deploy"
            ;;
    esac
)