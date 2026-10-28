"""pyhxt on the command line: a surface file in, a tetrahedral mesh file out (through meshio).

    pyhxt part.stl part.vtu --size 0.05
    python -m pyhxt part.stl part.vtu --size 0.05 --hole 0 0 0 --threads 8
"""
import argparse
import logging
import sys


def main(argv=None):
    import pyhxt
    p = argparse.ArgumentParser(
        prog='pyhxt', description='Fill a closed triangle surface with tetrahedra (HXT).',
        epilog='Files are read and written with meshio: STL, OBJ, PLY, VTK/VTU, MSH, XDMF, ...')
    p.add_argument('input', help='the closed surface')
    p.add_argument('output', help='the tetrahedral mesh; its suffix sets the format')
    p.add_argument('--size', type=float, help='largest edge length (default: from the surface)')
    p.add_argument('--size-min', type=float, default=0., help='smallest edge length')
    p.add_argument('--hole', type=float, nargs=3, action='append', metavar=('X', 'Y', 'Z'),
                   help='a point inside a region to leave empty (repeatable)')
    p.add_argument('--markers', default='markers',
                   help="the input's triangle data to keep as markers (default: markers)")
    p.add_argument('--boundary', metavar='FILE', help='also write the boundary triangles here')
    p.add_argument('--threads', type=int, default=0, help='0 (the default): every core')
    p.add_argument('--quality', type=float, default=.3,
                   help='improve tets below this quality (default 0.3)')
    p.add_argument('--no-optimize', action='store_true', help='skip the improvement')
    p.add_argument('--nonreproducible', action='store_true',
                   help='faster on large meshes, but the mesh changes from run to run')
    p.add_argument('-v', '--verbose', action='store_true', help="show HXT's messages")
    a = p.parse_args(argv)

    try:
        import meshio
    except ImportError:
        p.error('reading and writing files needs meshio: pip install meshio')
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format='%(message)s')
    from pyhxt._surface import surface
    points, triangles, markers = surface(meshio.read(a.input), markers=a.markers)
    try:
        mesh = pyhxt.tetrahedralize(
            points, triangles, markers, holes=a.hole, size=a.size, size_min=a.size_min,
            threads=a.threads, quality=a.quality, optimize=not a.no_optimize,
            reproducible=not a.nonreproducible)
    except (ValueError, pyhxt.HXTError) as e:
        sys.exit(f'pyhxt: {a.input}: {e}')
    mesh.write(a.output)
    if a.boundary:
        mesh.write(a.boundary, boundary=True)
    print(mesh.report())
    print(f'  wrote {a.output}' + (f' and {a.boundary}' if a.boundary else ''))


if __name__ == '__main__':
    main()
