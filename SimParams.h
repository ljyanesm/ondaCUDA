class cDataShot
{
    public:
        int id;
        int nUsedRecs;
        int nUsedRecLines;

        cDataShot()
        {
            id = -1;
            nUsedRecs = 0;
            nUsedRecLines = 0;
        }
};

typedef struct SimParams{
  float dx2;
  float dy2;
  float dz2;
  float dt2;
}SimParams;

typedef struct ThreadData{
  char* relFile;
  int volumeSize;
  int iniShot;
  int nShots;
  cDataShot *usedShots;
  int usedRecsBySource;
  int device;
  float *h_c;
  float xo, xf, yo, yf;
  float depth;
  SimParams h_params;
  int nx, ny, nz, nt;
  int nsamp, nrecs;
  int idsnap;
  int idt;
  int cuda;
  int *ixsource, *iysource, *izsource;
  int *ixrec, *iyrec, *izrec;
  int *nRecsByLine;
  float *source;
  int nSource;
  int nLinesR;
  float ***c;
  int numsnaps;
  float dx, dy, dz;
  float time1;
  int ddamp;
  int wdamp;
  int id;

}ThreadData;
