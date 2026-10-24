"""Model families declared as optional dependencies of the MedMLX distribution."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

FAMILIES = ("segment", "generate", "vision")


@dataclass(frozen=True, slots=True)
class SubPackage:
    """One optional runtime requirement and its installed version, if present."""

    name: str
    source: str
    version: str | None


@dataclass(frozen=True, slots=True)
class Family:
    """A named install extra and the runtimes declared in its requirements."""

    name: str
    version: str | None
    packages: tuple[SubPackage, ...]


def families() -> list[Family]:
    """Read family membership from installed metadata, including absent runtimes."""
    package = distribution("medmlx")
    requirements = [Requirement(text) for text in package.requires or []]
    tree: list[Family] = []
    for name in FAMILIES:
        members = [
            requirement
            for requirement in requirements
            if requirement.marker is not None
            and "extra" in str(requirement.marker)
            and requirement.marker.evaluate({"extra": name})
        ]
        tree.append(
            Family(
                name,
                package.version,
                tuple(
                    SubPackage(
                        item.name,
                        item.url or str(item.specifier),
                        _installed_version(item.name),
                    )
                    for item in sorted(members, key=lambda item: canonicalize_name(item.name))
                ),
            )
        )
    return tree


def family_index(tree: list[Family]) -> dict[str, str]:
    """Map each runtime's canonical distribution name to its install family."""
    return {
        canonicalize_name(package.name): family.name
        for family in tree
        for package in family.packages
    }


def _installed_version(name: str) -> str | None:
    try:
        return distribution(name).version
    except PackageNotFoundError:
        return None
