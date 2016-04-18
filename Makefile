CC = g++-4.9

ifeq ($(shell sw_vers 2>/dev/null | grep Mac | awk '{ print $$2}'),Mac)
	CFLAGS = -O3 -fopenmp -DOSX -I./include/
	LDFLAGS = -L./library -lfreeimage
else
	CFLAGS = -O3 -fopenmp
	LDFLAGS = -lfreeimage
endif


all : src/raytracer

src/raytracer : src/raytracer.cpp
	${CC} ${CFLAGS} src/raytracer.cpp -o raytracer ${LDFLAGS}

clean :
	rm -rf *.o src/*.o *~ raytracer