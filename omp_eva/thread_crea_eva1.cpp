#include<stdlib.h>
#include<omp.h>
#include<iostream>
#include<vector>

using namespace std;
#define MULTITHREAD 4
#define MULTICAND 32
#define LOOP 100

void test(bool printornot) {
    double begintime, endtime, timetaken = 0;
    double restime[MULTITHREAD] = {0};
    int i;
    
    for (i = 0; i < LOOP; i++) {
        std::vector<std::vector<std::pair<int, int>>> global_cand(MULTITHREAD);
        
        begintime = omp_get_wtime();

        #pragma omp parallel for num_threads(MULTITHREAD)
        for (int i = 0; i < MULTICAND; i++){
            double itime = omp_get_wtime();

            int threadid = omp_get_thread_num();
            for (int j = 0; j < 100; j++) {
                global_cand[threadid].push_back(make_pair(j, j+1));
            }
            

            double ftime = omp_get_wtime();
            restime[threadid] += (ftime - itime) * 1e6;
        }

        endtime = omp_get_wtime();
        timetaken += (endtime - begintime) * 1e6;
    }

    if (printornot) {
        timetaken /= LOOP;
        double maxtime = 0;
        for (int i = 0; i < MULTITHREAD; i++) {
            cout << restime[i] / LOOP << endl;
            maxtime = max(maxtime, restime[i] / LOOP);
        }
        cout << "Parallel: " << timetaken << endl;
        cout << "Diff: " << timetaken - maxtime << endl;
    }
}

int main() {
    test(false);
    test(true);
    return 0;
}