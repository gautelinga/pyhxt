/* pyhxt: HXT's hxtTetMesh called as gmsh 4.13.1's meshGRegionHxt.cpp calls it, on arrays,
 * with pyhxt's switches (pyhxt_flags.h) on top. Built against gmsh 4.13.1's contrib/hxt
 * sources. AGPL-3.0-or-later. */
#include <float.h>
#include <stdio.h>
#include <stdint.h>
#include "hxt_tetMesh.h"
#include "hxt_tools.h"
#include "predicates.h"
#include "pyhxt_flags.h"

int pyhxt_flags = 0;
uint32_t pyhxt_hole_first = 0, pyhxt_num_holes = 0;
uint64_t* pyhxt_tri2tet = NULL;

/* a size field: sets pts[4 i + 3] for each of the n points (x, y, z, interpolated size);
 * returns 0, or non-zero to stop meshing */
typedef int (*pyhxt_field_fn)(double *pts, uint64_t n);
/* HXT's messages: level 0 info, 1 debug, 2 warning, 3 error, 4 trace */
typedef void (*pyhxt_message_fn)(int level, const char *text);

static FILE *logfile = NULL;
static pyhxt_message_fn message = NULL;

static HXTStatus messageCallback(HXTMessage *msg)
{
  if(!msg) return HXT_STATUS_OK;
  if(logfile) {
    const char *level[] = {"Info", "Debug", "Warning", "Error", "Trace"};
    fprintf(logfile, "%s: %s\n", level[msg->level], msg->string);
    fflush(logfile);
  }
  if(message) message((int)msg->level, msg->string);
  return HXT_STATUS_OK;
}

typedef struct {
  double lcGlob;
  pyhxt_field_fn field;
} SizeData;

/* gmsh's nodalSizesCallBack with no size field, point size or entity size: the
 * background size is MAX_LC, so a point keeps its interpolated size capped by the
 * model's characteristic length, or takes that length where it has none. A user's field
 * then lowers it where it asks for smaller. */
static HXTStatus nodalSizes(double *pts, uint32_t *volume, size_t n, void *data)
{
  SizeData *d = data;
  HXT_INFO("Computing interpolated mesh sizes...");
  for(size_t i = 0; i < n; i++) {
    double s = pts[4 * i + 3];
    pts[4 * i + 3] = s > 0.0 ? (d->lcGlob < s ? d->lcGlob : s) : d->lcGlob;
  }
  if(d->field && d->field(pts, n))
    return HXT_ERROR_MSG(HXT_STATUS_ERROR, "the size field failed");
  HXT_INFO("Done computing interpolated mesh sizes");
  return HXT_STATUS_OK;
}

/* The version of this interface: 2 added nholes; 3 the point sizes, the size field and
 * the message callback, and pyhxt_locate. */
int pyhxt_abi(void) { return 3; }

/* xyz: nv x 3, the surface's vertices, and sizes: nv target sizes there (0: from the
 * surface's edges), or NULL; seeds: nseed x 3, embedded points, which follow them as
 * vertices nv, nv + 1, ..., the first nholes of them hole points (the volumes holding
 * them are left out); tri: ntri x 3 with one color each; volumes: nvol, each bounded by
 * the surfaces surfaces[offsets[i]:offsets[i+1]], or nvol 0 for every volume HXT finds;
 * flags: pyhxt_flags.h (0: as gmsh); field and messages: callbacks, or NULL. On success
 * *out holds HXT's mesh. */
int pyhxt_tetmesh(uint32_t nv, const double *xyz, const double *sizes, uint32_t nseed,
                  const double *seeds, uint32_t nholes, uint64_t ntri, const uint32_t *tri,
                  const uint32_t *color, uint32_t nvol, const uint32_t *offsets,
                  const uint32_t *surfaces, int threads, double sizeMin, double sizeMax,
                  double lcGlob, double tolerance, double qualityMin, int optimize,
                  int flags, pyhxt_field_fn field, pyhxt_message_fn messages,
                  const char *logpath, void **out)
{
  HXTMesh *m = NULL;
  SizeData sizeData = {lcGlob, field};
  *out = NULL;
  logfile = logpath ? fopen(logpath, "a") : NULL;
  message = messages;
  hxtSetMessageCallback(messageCallback);
  HXTStatus status = hxtMeshCreate(&m);
  if(status != HXT_STATUS_OK) goto done;

  pyhxt_flags = flags;
  pyhxt_hole_first = nv;
  pyhxt_num_holes = nholes < nseed ? nholes : nseed;
  m->vertices.num = m->vertices.size = nv + nseed;
  if((status = hxtAlignedMalloc(&m->vertices.coord,
                                4 * ((size_t)nv + nseed) * sizeof(double))))
    goto done;
  for(uint32_t i = 0; i < nv + nseed; i++) {
    const double *x = i < nv ? xyz + 3 * (size_t)i : seeds + 3 * (size_t)(i - nv);
    m->vertices.coord[4 * i + 0] = x[0];
    m->vertices.coord[4 * i + 1] = x[1];
    m->vertices.coord[4 * i + 2] = x[2];
    m->vertices.coord[4 * i + 3] = sizes && i < nv && sizes[i] > 0.0 ? sizes[i] : 0.0;
  }
  if(nseed) {
    m->points.num = m->points.size = nseed;
    if((status = hxtAlignedMalloc(&m->points.node, nseed * sizeof(uint32_t))) ||
       (status = hxtAlignedMalloc(&m->points.color, nseed * sizeof(uint32_t))))
      goto done;
    for(uint32_t i = 0; i < nseed; i++) {
      m->points.node[i] = nv + i;
      m->points.color[i] = 0;
    }
  }
  m->triangles.num = m->triangles.size = ntri;
  if((status = hxtAlignedMalloc(&m->triangles.node, 3 * ntri * sizeof(uint32_t))) ||
     (status = hxtAlignedMalloc(&m->triangles.color, ntri * sizeof(uint32_t))))
    goto done;
  for(uint64_t i = 0; i < 3 * ntri; i++) m->triangles.node[i] = tri[i];
  for(uint64_t i = 0; i < ntri; i++) m->triangles.color[i] = color[i];

  m->brep.numVolumes = nvol;
  if(nvol == 0) goto options; /* HXT finds the volumes and fills the brep itself */
  if((status = hxtAlignedMalloc(&m->brep.numSurfacesPerVolume, nvol * sizeof(uint32_t))) ||
     (status = hxtAlignedMalloc(&m->brep.surfacesPerVolume,
                                offsets[nvol] * sizeof(uint32_t))))
    goto done;
  for(uint32_t i = 0; i < nvol; i++)
    m->brep.numSurfacesPerVolume[i] = offsets[i + 1] - offsets[i];
  for(uint32_t i = 0; i < offsets[nvol]; i++) m->brep.surfacesPerVolume[i] = surfaces[i];
  /* unused by hxtTetMesh, but in its input hash: gmsh gives the number of surfaces and
   * no curves on them (a volume's surfaces are distinct, and one volume per call here) */
  m->brep.numSurfaces = offsets[nvol];
  if((status = hxtAlignedMalloc(&m->brep.numCurvesPerSurface,
                                offsets[nvol] * sizeof(uint32_t))))
    goto done;
  for(uint32_t i = 0; i < offsets[nvol]; i++) m->brep.numCurvesPerSurface[i] = 0;

options:;
  HXTTetMeshOptions options = {
    threads, threads, threads, // default, Delaunay and improvement threads
    !(flags & PYHXT_NONREPRODUCIBLE), // reproducible
    1, // verbosity, as gmsh's at General.Verbosity 5
    1, // stat
    1, // refine
    optimize,
    tolerance,
    {NULL, NULL, qualityMin},
    {nodalSizes, &sizeData, sizeMin, sizeMax, 1.0}};
  status = hxtTetMesh(m, &options);

done:
  pyhxt_flags = 0;
  pyhxt_num_holes = 0;
  if(status == HXT_STATUS_OK) {
    /* the adjacency (32 B a tet) and flags are not handed out: free them before the caller
     * allocates its copies, so that HXT's mesh and those copies are not all held at once */
    hxtAlignedFree(&m->tetrahedra.neigh);
    hxtAlignedFree(&m->tetrahedra.flag);
    *out = m;
  }
  else if(m)
    hxtMeshDelete(&m);
  if(logfile) fclose(logfile);
  logfile = NULL;
  message = NULL;
  return (int)status;
}

void pyhxt_counts(void *mesh, uint32_t *nv, uint64_t *ntet, uint64_t *ntri)
{
  HXTMesh *m = mesh;
  *nv = m->vertices.num;
  *ntet = m->tetrahedra.num;
  *ntri = m->triangles.num;
}

/* coords: nv x 3; tets: ntet x 4 and their colors; triangles: ntri x 3 */
void pyhxt_copy(void *mesh, double *coords, uint32_t *tets, uint32_t *tetcolor,
                  uint32_t *tris)
{
  HXTMesh *m = mesh;
  for(uint32_t i = 0; i < m->vertices.num; i++)
    for(int j = 0; j < 3; j++) coords[3 * i + j] = m->vertices.coord[4 * i + j];
  for(uint64_t i = 0; i < 4 * m->tetrahedra.num; i++) tets[i] = m->tetrahedra.node[i];
  for(uint64_t i = 0; i < m->tetrahedra.num; i++) tetcolor[i] = m->tetrahedra.color[i];
  for(uint64_t i = 0; i < 3 * m->triangles.num; i++) tris[i] = m->triangles.node[i];
}

void pyhxt_free(void *mesh)
{
  HXTMesh *m = mesh;
  if(m) hxtMeshDelete(&m);
}

/* colors: ntri, the surface number of each of HXT's triangles */
void pyhxt_copy_triangle_colors(void *mesh, uint32_t *colors)
{
  HXTMesh *m = mesh;
  for(uint64_t i = 0; i < m->triangles.num; i++)
    colors[i] = m->triangles.color ? m->triangles.color[i] : 0;
}

/* the color of the meshed tet holding each of the n points xyz (on its boundary counts),
 * or UINT32_MAX for a point in no meshed tet; a scan of all tets per point, with exact
 * orientation tests, in parallel */
void pyhxt_locate(void *mesh, uint32_t n, const double *xyz, uint32_t *color)
{
  HXTMesh *m = mesh;
  const double *X = m->vertices.coord;
  for(uint32_t p = 0; p < n; p++) {
    const double *q = xyz + 3 * (size_t)p;
    uint64_t found = UINT64_MAX;
    #pragma omp parallel for reduction(min:found) schedule(static)
    for(uint64_t t = 0; t < m->tetrahedra.num; t++) {
      const uint32_t *v = m->tetrahedra.node + 4 * t;
      if(v[3] == HXT_GHOST_VERTEX || m->tetrahedra.color[t] == HXT_COLOR_OUT) continue;
      double a[3], b[3], c[3], d[3];
      for(int k = 0; k < 3; k++) {
        a[k] = X[4 * v[0] + k]; b[k] = X[4 * v[1] + k];
        c[k] = X[4 * v[2] + k]; d[k] = X[4 * v[3] + k];
      }
      double o = orient3d(a, b, c, d);
      double o0 = orient3d(q, b, c, d), o1 = orient3d(a, q, c, d);
      double o2 = orient3d(a, b, q, d), o3 = orient3d(a, b, c, q);
      if(o > 0 ? (o0 >= 0 && o1 >= 0 && o2 >= 0 && o3 >= 0)
               : (o0 <= 0 && o1 <= 0 && o2 <= 0 && o3 <= 0))
        found = t < found ? t : found;
    }
    color[p] = found == UINT64_MAX ? UINT32_MAX : m->tetrahedra.color[found];
  }
}
