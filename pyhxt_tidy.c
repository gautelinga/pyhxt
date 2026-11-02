/* pyhxt_tidy.c: two passes over a finished mesh, for tets HXT's own optimiser cannot
 * improve because it may not change the surface.
 *
 * pyhxt_peel_flat: a tet of no volume whose four points are all on the surface, two of its
 * faces being adjacent surface triangles in one plane (HXT's boundary recovery leaves one
 * now and then). Its interior edge holds an unknown in a cell of no volume. The triangles'
 * shared diagonal is swapped for the tet's other two faces and the tet removed: the surface
 * is the same set of points. Triangles with a protected color are never touched.
 *
 * pyhxt_relax_slivers: the free (not surface) vertices of the tets within some angle of
 * flat are moved to where the worst tet of their star is furthest from flat, keeping every
 * tet of the star the same way round; vertices allowed to slide along a sphere move in its
 * tangent plane and are put back on it. Gauss-Seidel over the vertices, worst first, a few
 * sweeps; a move is kept only if it improves its star. The search is Nelder-Mead exactly as
 * scipy.optimize's (not adaptive, rho 1, chi 2, psi 0.5, sigma 0.5), from an initial
 * simplex 0.2 times the star's size, xatol 1e-4 of it, fatol 1e-3 degrees: this is
 * mesh-sphere-packing's hxtmesh.relax_slivers (numba and scipy) in C, the stars found
 * through an index rather than a scan.
 *
 * The measure of a tet is the angle (degrees) by which it falls short of flat: the smaller
 * of its smallest dihedral angle and 180 less its largest.
 *
 * AGPL-3.0-or-later. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RAD2DEG (180.0 / 3.141592653589793)

/* the (unnormalised) normal of triangle abc and its length */
static inline void normal(const double *X, int64_t a, int64_t b, int64_t c, double *n)
{
  double ux = X[3 * b] - X[3 * a], uy = X[3 * b + 1] - X[3 * a + 1], uz = X[3 * b + 2] - X[3 * a + 2];
  double wx = X[3 * c] - X[3 * a], wy = X[3 * c + 1] - X[3 * a + 1], wz = X[3 * c + 2] - X[3 * a + 2];
  n[0] = uy * wz - uz * wy;
  n[1] = uz * wx - ux * wz;
  n[2] = ux * wy - uy * wx;
  n[3] = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
}

/* one tet's distance from flat and its signed volume */
static double tet_quality(const double *X, const int64_t *t, double *vol)
{
  double n[4][4];
  normal(X, t[1], t[2], t[3], n[0]);
  normal(X, t[0], t[2], t[3], n[1]);
  normal(X, t[0], t[1], t[3], n[2]);
  normal(X, t[0], t[1], t[2], n[3]);
  *vol = (n[3][0] * (X[3 * t[3]] - X[3 * t[0]]) + n[3][1] * (X[3 * t[3] + 1] - X[3 * t[0] + 1])
          + n[3][2] * (X[3 * t[3] + 2] - X[3 * t[0] + 2])) / 6.;
  /* the dihedral 180 - acos(c) grows with the cosine c, so the smallest and largest come
   * from the smallest and largest cosines: two acos, not six, and the same numbers */
  double cmin = 2., cmax = -2.;
  for(int i = 0; i < 4; i++) {
    if(n[i][3] == 0.) return 0.;
    for(int j = i + 1; j < 4; j++) {
      double c = (n[i][0] * n[j][0] + n[i][1] * n[j][1] + n[i][2] * n[j][2]) / (n[i][3] * n[j][3]);
      c = c < -1. ? -1. : c;
      c = 1. < c ? 1. : c;
      cmin = c < cmin ? c : cmin;
      cmax = cmax < c ? c : cmax;
    }
  }
  double lo = 180. - acos(cmin) * RAD2DEG, hi = 180. - acos(cmax) * RAD2DEG;
  return lo <= 180. - hi ? lo : 180. - hi;
}

/* every tet's measure and signed volume (either may be NULL) */
void pyhxt_quality(uint64_t nt, const double *X, const int64_t *T, double *q, double *vol)
{
#pragma omp parallel for schedule(static)
  for(int64_t k = 0; k < (int64_t)nt; k++) {
    double v, m = tet_quality(X, T + 4 * k, &v);
    if(q) q[k] = m;
    if(vol) vol[k] = v;
  }
}

/* ---------------------------------------------------------------- peeling */

typedef struct { int64_t a, b, c, row; } Face;

static int face_cmp(const void *p, const void *q)
{
  const Face *x = p, *y = q;
  if(x->a != y->a) return x->a < y->a ? -1 : 1;
  if(x->b != y->b) return x->b < y->b ? -1 : 1;
  if(x->c != y->c) return x->c < y->c ? -1 : 1;
  return 0;
}

static Face sorted_face(int64_t a, int64_t b, int64_t c, int64_t row)
{
  int64_t t;
  if(b < a) { t = a; a = b; b = t; }
  if(c < b) { t = b; b = c; c = t; }
  if(b < a) { t = a; a = b; b = t; }
  return (Face){a, b, c, row};
}

/* T (nt x 4) and tri (ntri x 3, colors color) are changed in place: the peeled tets are
 * moved to the end of T and each changed triangle row rewritten (same count, same color,
 * oriented as the one it replaces). flat: 6|V| below flat times the longest edge cubed.
 * protect: nprotect colors never touched. removed (nt, or NULL): set to 1 at the peeled
 * tets' original rows, 0 elsewhere. Returns the number peeled (the tets kept are the first
 * nt minus that), or -1 if out of memory. */
int64_t pyhxt_peel_flat(uint64_t nv, const double *X, uint64_t nt, int64_t *T, uint64_t ntri,
                        int64_t *tri, const int64_t *color, double flat,
                        uint64_t nprotect, const int64_t *protect, uint8_t *removed)
{
  if(removed) memset(removed, 0, nt);
  uint8_t *on = calloc(nv, 1);
  if(!on) return -1;
  for(uint64_t i = 0; i < 3 * ntri; i++) on[tri[i]] = 1;
  /* the candidates: all four points on the surface and flat */
  int64_t ncand = 0, cap = 64, *cand = malloc(cap * sizeof(int64_t));
  if(!cand) { free(on); return -1; }
  for(uint64_t k = 0; k < nt; k++) {
    const int64_t *t = T + 4 * k;
    if(!(on[t[0]] && on[t[1]] && on[t[2]] && on[t[3]])) continue;
    double v;
    tet_quality(X, t, &v);
    double e = 0.;
    for(int a = 0; a < 4; a++)
      for(int b = a + 1; b < 4; b++) {
        double dx = X[3 * t[a]] - X[3 * t[b]], dy = X[3 * t[a] + 1] - X[3 * t[b] + 1],
               dz = X[3 * t[a] + 2] - X[3 * t[b] + 2];
        double l = sqrt(dx * dx + dy * dy + dz * dz);
        e = l > e ? l : e;
      }
    if(!(fabs(6. * v) < flat * e * e * e)) continue;
    if(ncand == cap) {
      int64_t *c2 = realloc(cand, 2 * cap * sizeof(int64_t));
      if(!c2) { free(on); free(cand); return -1; }
      cand = c2; cap *= 2;
    }
    cand[ncand++] = (int64_t)k;
  }
  free(on);
  if(!ncand) { free(cand); return 0; }

  Face *faces = malloc(ntri * sizeof(Face));
  uint8_t *drop = calloc(nt, 1);
  if(!faces || !drop) { free(cand); free(faces); free(drop); return -1; }
  for(uint64_t i = 0; i < ntri; i++)
    faces[i] = sorted_face(tri[3 * i], tri[3 * i + 1], tri[3 * i + 2], (int64_t)i);
  qsort(faces, ntri, sizeof(Face), face_cmp);

  int64_t peeled = 0;
  for(int64_t ci = 0; ci < ncand; ci++) {
    const int64_t *t = T + 4 * cand[ci];
    int64_t rows[2], other[2][3], nrow = 0, nother = 0;
    for(int k = 0; k < 4; k++) {
      int64_t f[3], m = 0;
      for(int j = 0; j < 4; j++)
        if(j != k) f[m++] = t[j];
      Face key = sorted_face(f[0], f[1], f[2], 0);
      Face *hit = bsearch(&key, faces, ntri, sizeof(Face), face_cmp);
      if(hit) {
        if(nrow < 2) rows[nrow] = hit->row;
        nrow++;
      }
      else {
        /* as mesh-sphere-packing wrote it: the corners in ascending order, then oriented */
        if(nother < 2) {
          other[nother][0] = key.a; other[nother][1] = key.b; other[nother][2] = key.c;
        }
        nother++;
      }
    }
    if(nrow != 2 || color[rows[0]] != color[rows[1]]) continue;
    int skip = 0;
    for(uint64_t p = 0; p < nprotect; p++) skip |= color[rows[0]] == protect[p];
    if(skip) continue;
    for(int r = 0; r < 2; r++) {
      int64_t *old = tri + 3 * rows[r], *f = other[r];
      double n_old[4], n_new[4];
      normal(X, old[0], old[1], old[2], n_old);
      normal(X, f[0], f[1], f[2], n_new);
      if(n_new[0] * n_old[0] + n_new[1] * n_old[1] + n_new[2] * n_old[2] < 0.) {
        int64_t s = f[1]; f[1] = f[2]; f[2] = s;
      }
      /* the face index follows the row */
      Face key = sorted_face(old[0], old[1], old[2], 0);
      Face *hit = bsearch(&key, faces, ntri, sizeof(Face), face_cmp);
      memcpy(old, f, 3 * sizeof(int64_t));
      if(hit) {
        *hit = sorted_face(f[0], f[1], f[2], rows[r]);
        qsort(faces, ntri, sizeof(Face), face_cmp);  /* rare: a handful a mesh */
      }
    }
    drop[cand[ci]] = 1;
    if(removed) removed[cand[ci]] = 1;
    peeled++;
  }
  /* the kept tets first, in their order */
  if(peeled) {
    int64_t *gone = malloc(4 * peeled * sizeof(int64_t));
    if(!gone) { free(cand); free(faces); free(drop); return -1; }
    uint64_t w = 0, g = 0;
    for(uint64_t k = 0; k < nt; k++) {
      if(drop[k]) memcpy(gone + 4 * g++, T + 4 * k, 4 * sizeof(int64_t));
      else {
        if(w != k) memcpy(T + 4 * w, T + 4 * k, 4 * sizeof(int64_t));
        w++;
      }
    }
    memcpy(T + 4 * w, gone, 4 * peeled * sizeof(int64_t));
    free(gone);
  }
  free(cand); free(faces); free(drop);
  return peeled;
}

/* ---------------------------------------------------------------- slivers */

/* the length of a 3-vector as numpy's np.linalg.norm gives it, sqrt(x.dot(x)) through BLAS,
 * whose dot fuses its multiply-adds in this order (checked bit for bit on numpy 2.5) */
static inline double norm3(const double *u)
{
  return sqrt(fma(u[2], u[2], fma(u[1], u[1], u[0] * u[0])));
}

typedef struct {
  double *X;
  const int64_t *star;  /* nstar x 4 */
  int64_t nstar, v;
  const double *sign;
  /* sliding: the sphere and the tangent frame at the start */
  int slide;
  double c[3], R, x0[3], e1[3], e2[3];
} Star;

static void on_sphere(const Star *s, const double *y, double *x)
{
  double u[3];
  for(int i = 0; i < 3; i++) u[i] = s->x0[i] - s->c[i] + y[0] * s->e1[i] + y[1] * s->e2[i];
  double r = norm3(u);
  for(int i = 0; i < 3; i++) x[i] = s->c[i] + s->R * u[i] / r;
}

/* minus the star's worst measure with v at x, or infinity if a tet turns inside out */
static double star_cost(const Star *s, const double *y)
{
  double x[3];
  if(s->slide) on_sphere(s, y, x);
  else memcpy(x, y, sizeof x);
  memcpy(s->X + 3 * s->v, x, sizeof x);
  double worst = 180.;
  for(int64_t k = 0; k < s->nstar; k++) {
    double vol, q = tet_quality(s->X, s->star + 4 * k, &vol);
    if(vol * s->sign[k] <= 0.) return INFINITY;
    worst = q < worst ? q : worst;
  }
  return -worst;
}

/* scipy's argsort of a few values: numpy sorts so short an array by insertion, stably */
static void sort_simplex(int n, double *sim, double *fsim)
{
  for(int i = 1; i <= n; i++) {
    double f = fsim[i], x[3];
    memcpy(x, sim + 3 * i, sizeof x);
    int j = i - 1;
    while(j >= 0 && f < fsim[j]) {
      fsim[j + 1] = fsim[j];
      memcpy(sim + 3 * (j + 1), sim + 3 * j, sizeof x);
      j--;
    }
    fsim[j + 1] = f;
    memcpy(sim + 3 * (j + 1), x, sizeof x);
  }
}

/* scipy.optimize's Nelder-Mead (adaptive=False) from the simplex sim ((n+1) x 3 rows, n
 * used), at most maxfev evaluations; x gets the best point, the result is its value */
static double nelder_mead(const Star *s, int n, double *sim, double xatol, double fatol,
                          int maxfev, double *x)
{
  const double rho = 1., chi = 2., psi = .5, sigma = .5;
  double fsim[4];
  int calls = 0, exhausted = 0;
#define EVAL(p, out) do { if(calls >= maxfev) { exhausted = 1; } else { calls++; out = star_cost(s, p); } } while(0)
  for(int k = 0; k <= n; k++) fsim[k] = INFINITY;
  for(int k = 0; k <= n && !exhausted; k++) EVAL(sim + 3 * k, fsim[k]);
  sort_simplex(n, sim, fsim);
  while(calls < maxfev && !exhausted) {
    double dx = 0., df = 0.;
    for(int k = 1; k <= n; k++)
      for(int i = 0; i < n; i++) {
        double d = fabs(sim[3 * k + i] - sim[i]);
        dx = d > dx || isnan(d) ? d : dx;
      }
    int nan_f = 0;
    for(int k = 1; k <= n; k++) {
      double d = fabs(fsim[0] - fsim[k]);
      if(isnan(d)) nan_f = 1;
      df = d > df ? d : df;
    }
    if(!isnan(dx) && dx <= xatol && !nan_f && df <= fatol) break;
    double xbar[3] = {0., 0., 0.}, xr[3], xe[3], xc[3], fxr = 0., fxe = 0., fxc = 0.;
    for(int k = 0; k < n; k++)
      for(int i = 0; i < n; i++) xbar[i] += sim[3 * k + i];
    for(int i = 0; i < n; i++) xbar[i] /= n;
    double *last = sim + 3 * n;
    for(int i = 0; i < n; i++) xr[i] = (1 + rho) * xbar[i] - rho * last[i];
    EVAL(xr, fxr);
    if(exhausted) break;
    int shrink = 0;
    if(fxr < fsim[0]) {
      for(int i = 0; i < n; i++) xe[i] = (1 + rho * chi) * xbar[i] - rho * chi * last[i];
      EVAL(xe, fxe);
      if(exhausted) break;
      if(fxe < fxr) { memcpy(last, xe, sizeof xe); fsim[n] = fxe; }
      else { memcpy(last, xr, sizeof xr); fsim[n] = fxr; }
    }
    else if(fxr < fsim[n - 1]) { memcpy(last, xr, sizeof xr); fsim[n] = fxr; }
    else {
      if(fxr < fsim[n]) {
        for(int i = 0; i < n; i++) xc[i] = (1 + psi * rho) * xbar[i] - psi * rho * last[i];
        EVAL(xc, fxc);
        if(exhausted) break;
        if(fxc <= fxr) { memcpy(last, xc, sizeof xc); fsim[n] = fxc; }
        else shrink = 1;
      }
      else {
        for(int i = 0; i < n; i++) xc[i] = (1 - psi) * xbar[i] + psi * last[i];
        EVAL(xc, fxc);
        if(exhausted) break;
        if(fxc < fsim[n]) { memcpy(last, xc, sizeof xc); fsim[n] = fxc; }
        else shrink = 1;
      }
      if(shrink)
        for(int j = 1; j <= n; j++) {
          for(int i = 0; i < n; i++) sim[3 * j + i] = sim[i] + sigma * (sim[3 * j + i] - sim[i]);
          EVAL(sim + 3 * j, fsim[j]);
          if(exhausted) break;
        }
    }
    sort_simplex(n, sim, fsim);
  }
  /* scipy sorts after an exhausted iteration too */
  sort_simplex(n, sim, fsim);
#undef EVAL
  memcpy(x, sim, 3 * sizeof(double));
  double best = fsim[0];
  for(int k = 1; k <= n; k++) best = fsim[k] < best ? fsim[k] : best;
  return best;
}

typedef struct { double worst; int64_t v; } Rank;

static int rank_cmp(const void *p, const void *q)
{
  const Rank *a = p, *b = q;
  if(a->worst != b->worst) return a->worst < b->worst ? -1 : 1;
  return a->v < b->v ? -1 : (a->v > b->v);
}

static int i64_cmp(const void *p, const void *q)
{
  int64_t a = *(const int64_t *)p, b = *(const int64_t *)q;
  return a < b ? -1 : (a > b);
}

/* X (nv x 3) is changed in place; q (nt) is every tet's measure on entry (pyhxt_quality)
 * and on return. fixed: surface vertices; slide (or NULL): those of them that may move along
 * the sphere centre/radius. stats: tets below angle before and after, vertices slid.
 * Returns 0, or -1 if out of memory. */
int pyhxt_relax_slivers(uint64_t nv, double *X, uint64_t nt, const int64_t *T, double *q,
                        const uint8_t *fixed, const uint8_t *slide, const double *centre,
                        const double *radius, double angle, int sweeps, int maxfev,
                        int64_t *stats)
{
  int64_t before = 0;
  for(uint64_t k = 0; k < nt; k++) before += q[k] < angle;
  uint8_t *isfree = calloc(nv, 1), *slid = calloc(nv, 1);
  double *worst = malloc(nv * sizeof(double));
  if(!isfree || !slid || !worst) { free(isfree); free(slid); free(worst); return -1; }
  int status = 0;
  for(int sweep = 0; sweep < sweeps; sweep++) {
    /* the vertices to move: the free ones of the bad tets, and of the bad tets with every
     * vertex on the surface, those that may slide; worst first, then by number */
    int64_t nbad = 0;
    for(uint64_t k = 0; k < nt; k++) nbad += q[k] < angle;
    if(!nbad) break;
    memset(isfree, 0, nv);
    for(uint64_t i = 0; i < nv; i++) worst[i] = INFINITY;
    for(uint64_t k = 0; k < nt; k++) {
      if(!(q[k] < angle)) continue;
      const int64_t *t = T + 4 * k;
      int stuck = fixed[t[0]] && fixed[t[1]] && fixed[t[2]] && fixed[t[3]];
      for(int j = 0; j < 4; j++) {
        if(!fixed[t[j]] || (stuck && slide && slide[t[j]])) isfree[t[j]] = 1;
        worst[t[j]] = q[k] < worst[t[j]] ? q[k] : worst[t[j]];
      }
    }
    int64_t nfree = 0;
    for(uint64_t i = 0; i < nv; i++) nfree += isfree[i];
    if(!nfree) break;
    Rank *order = malloc(nfree * sizeof(Rank));
    if(!order) { status = -1; break; }
    nfree = 0;
    for(uint64_t i = 0; i < nv; i++)
      if(isfree[i]) order[nfree++] = (Rank){worst[i], (int64_t)i};
    qsort(order, nfree, sizeof(Rank), rank_cmp);

    /* the tets touching a vertex to move, in order, and each such vertex's star in them */
    int64_t nnear = 0;
    for(uint64_t k = 0; k < nt; k++) {
      const int64_t *t = T + 4 * k;
      nnear += isfree[t[0]] || isfree[t[1]] || isfree[t[2]] || isfree[t[3]];
    }
    int64_t *near = malloc(nnear * sizeof(int64_t)), *key = malloc(4 * nnear * 2 * sizeof(int64_t));
    if(!near || !key) { free(order); free(near); free(key); status = -1; break; }
    nnear = 0;
    for(uint64_t k = 0; k < nt; k++) {
      const int64_t *t = T + 4 * k;
      if(isfree[t[0]] || isfree[t[1]] || isfree[t[2]] || isfree[t[3]]) near[nnear++] = (int64_t)k;
    }
    /* (vertex, position in near) pairs sorted: a vertex's tets in near's order */
    for(int64_t r = 0; r < nnear; r++)
      for(int j = 0; j < 4; j++) {
        key[2 * (4 * r + j)] = T[4 * near[r] + j];
        key[2 * (4 * r + j) + 1] = r;
      }
    qsort(key, 4 * nnear, 2 * sizeof(int64_t), i64_cmp);  /* by vertex */
    /* within a vertex, by position (qsort is not stable) */
    for(int64_t a = 0; a < 4 * nnear;) {
      int64_t b = a;
      while(b < 4 * nnear && key[2 * b] == key[2 * a]) b++;
      if(b - a > 1) {
        int64_t *pos = malloc((b - a) * sizeof(int64_t));
        if(!pos) { status = -1; break; }
        for(int64_t m = a; m < b; m++) pos[m - a] = key[2 * m + 1];
        qsort(pos, b - a, sizeof(int64_t), i64_cmp);
        for(int64_t m = a; m < b; m++) key[2 * m + 1] = pos[m - a];
        free(pos);
      }
      a = b;
    }
    if(status) { free(order); free(near); free(key); break; }

    int64_t cap = 64;
    int64_t *star = malloc(4 * cap * sizeof(int64_t));
    double *sign = malloc(cap * sizeof(double));
    for(int64_t o = 0; o < nfree && star && sign; o++) {
      int64_t v = order[o].v;
      /* v's star: binary search for its run of pairs */
      int64_t lo = 0, hi = 4 * nnear;
      while(lo < hi) { int64_t m = (lo + hi) / 2; if(key[2 * m] < v) lo = m + 1; else hi = m; }
      int64_t a = lo;
      hi = 4 * nnear;
      while(lo < hi) { int64_t m = (lo + hi) / 2; if(key[2 * m] <= v) lo = m + 1; else hi = m; }
      int64_t ns = lo - a;
      if(ns > cap) {
        while(cap < ns) cap *= 2;
        int64_t *s2 = realloc(star, 4 * cap * sizeof(int64_t));
        double *g2 = realloc(sign, cap * sizeof(double));
        if(!s2 || !g2) { free(s2 ? s2 : star); free(g2 ? g2 : sign); star = NULL; sign = NULL; break; }
        star = s2; sign = g2;
      }
      double q0 = 180.;
      for(int64_t m = 0; m < ns; m++) {
        memcpy(star + 4 * m, T + 4 * near[key[2 * (a + m) + 1]], 4 * sizeof(int64_t));
        double vol, qq = tet_quality(X, star + 4 * m, &vol);
        sign[m] = vol > 0. ? 1. : (vol < 0. ? -1. : 0.);
        q0 = qq < q0 ? qq : q0;
      }
      double x0[3], scale = 0.;
      memcpy(x0, X + 3 * v, sizeof x0);
      for(int64_t m = 0; m < 4 * ns; m++) {
        int64_t w = star[m];
        double dx = X[3 * w] - x0[0], dy = X[3 * w + 1] - x0[1], dz = X[3 * w + 2] - x0[2];
        double l = sqrt(dx * dx + dy * dy + dz * dz);
        scale = l > scale ? l : scale;
      }
      Star s = {X, star, ns, v, sign, 0, {0}, 0., {0}, {0}, {0}};
      double sim[12], best[3], f;
      if(fixed[v]) {
        s.slide = 1;
        memcpy(s.c, centre + 3 * v, sizeof s.c);
        s.R = radius[v];
        memcpy(s.x0, x0, sizeof x0);
        double nn[3] = {(x0[0] - s.c[0]) / s.R, (x0[1] - s.c[1]) / s.R, (x0[2] - s.c[2]) / s.R};
        double ax[3] = {1., 0., 0.};
        if(!(fabs(nn[0]) < .9)) { ax[0] = 0.; ax[1] = 1.; }
        s.e1[0] = nn[1] * ax[2] - nn[2] * ax[1];
        s.e1[1] = nn[2] * ax[0] - nn[0] * ax[2];
        s.e1[2] = nn[0] * ax[1] - nn[1] * ax[0];
        double l = norm3(s.e1);
        for(int i = 0; i < 3; i++) s.e1[i] /= l;
        s.e2[0] = nn[1] * s.e1[2] - nn[2] * s.e1[1];
        s.e2[1] = nn[2] * s.e1[0] - nn[0] * s.e1[2];
        s.e2[2] = nn[0] * s.e1[1] - nn[1] * s.e1[0];
        memset(sim, 0, sizeof sim);
        sim[3] = .2 * scale;
        sim[7] = .2 * scale;
        f = nelder_mead(&s, 2, sim, 1e-4 * scale, 1e-3, maxfev, best);
        int better = isfinite(f) && -f > q0;
        if(better) { on_sphere(&s, best, X + 3 * v); slid[v] = 1; }
        else memcpy(X + 3 * v, x0, sizeof x0);
        continue;
      }
      for(int k = 0; k < 4; k++) memcpy(sim + 3 * k, x0, sizeof x0);
      for(int i = 0; i < 3; i++) sim[3 * (i + 1) + i] = x0[i] + .2 * scale;
      f = nelder_mead(&s, 3, sim, 1e-4 * scale, 1e-3, maxfev, best);
      memcpy(X + 3 * v, isfinite(f) && -f > q0 ? best : x0, sizeof x0);
    }
    if(!star || !sign) status = -1;
    free(star); free(sign);
    /* the measures of the tets that may have changed */
#pragma omp parallel for schedule(static)
    for(int64_t r = 0; r < nnear; r++) {
      double vol;
      q[near[r]] = tet_quality(X, T + 4 * near[r], &vol);
    }
    free(order); free(near); free(key);
    if(status) break;
  }
  int64_t after = 0, nslid = 0;
  for(uint64_t k = 0; k < nt; k++) after += q[k] < angle;
  for(uint64_t i = 0; i < nv; i++) nslid += slid[i];
  stats[0] = before; stats[1] = after; stats[2] = nslid;
  free(isfree); free(slid); free(worst);
  return status;
}
