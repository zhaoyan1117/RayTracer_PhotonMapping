#include <sstream>
#include <stdio.h>
#include <string>
#include <omp.h>
#include "RTtools.h"
#include "Photon.h"
#include "FreeImage.h"

using namespace std;


class Sampler {
  public:
    Point LU, LD, RU, RD;
    Sample cur;
    Point base;
    float widthP, heightP;
    Vector stepH, stepV;
    Vector microStepH, microStepV;

    float SPP; //Samples Per Pixel.
    float step;

    Sampler() {
        if (debug) cout << "Zero-parameter constructor of Sampler is called." << endl;
    }

    float sampleNum(){
        return widthP*heightP;
    }

    Sampler(Point lu, Point ld, Point ru, Point rd, float w, float h, float spp) {
        LU = lu; 
        LD = ld; 
        RU = ru; RD = rd;
        widthP = w;
        heightP = h;

        SPP = spp;
        step = sqrt(SPP);

        stepH = Vector::multiVS(Point::minusPoint(RU, LU), 1/widthP); //from left to right.
        stepV = Vector::multiVS(Point::minusPoint(LU, LD), 1/heightP); //from down to up.
        microStepH = Vector::multiVS(stepH, 1/(2*step));
        microStepV = Vector::multiVS(stepV, 1/(2*step));

        base = LD;

        cur = Sample(0, 0);
    }

    bool getSample(Sample* sample, std::vector<Point> *points) {
        if (cur.y == heightP) {
            return false;
        }
        sample->x = cur.x; sample->y = cur.y;
        
        Point start = Point::addVector(base, 
                        Vector::addVector(
                            Vector::multiVS(stepH, cur.x), 
                            Vector::multiVS(stepV, cur.y)
                        )
                    );

        Point curPoint;

        for (int i = 1; i < 2*step; i++) {
            for (int j = 1; j < 2*step; j++) {
                if (((i%2) != 0) and ((j%2) != 0)) {
                    curPoint = Point::addVector(start, 
                            Vector::addVector(
                                Vector::multiVS(microStepH, (float) i/2), 
                                Vector::multiVS(microStepV, (float) j/2)
                            )
                        );


                    // If SPP is not 1, jittering in both s and t directions.
                    if (SPP != 1) {
                        curPoint = Point::addVector(curPoint, 
                                Vector::addVector(
                                    Vector::multiVS(microStepH, generatRandom()), 
                                    Vector::multiVS(microStepV, generatRandom())
                                )
                            );
                    }

                    points->push_back(curPoint);
                }
            }
        }

        if (cur.x < widthP-1) {
            ++cur.x;
        } else {
            cur.x = 0;
            ++cur.y;
        }
        return true;
    }

    // Generate peudo-random numbers in [-1.0, 1.0].
    float generatRandom() {
        float r = (-0.2) + 0.4*(float)rand()/((float)RAND_MAX);
        return r;
    }
};

class Camera {
  public:
    Point pos;
    float clip;
    bool DOF;
    Vector upD;
    Vector viewD;
    Vector diskD;

    Camera() {
        if (debug) cout << "Zero-parameter constructor of Camera is called." << endl;
    }

    Camera(Point p, float c, Vector upd, Vector viewd, Vector diskd, bool dof) {
        pos = p; clip = c;
        upD = upd; viewD = viewd; diskD = diskd;
        DOF = dof;
    }

    void generateRay(Point& pixel, std::vector<Ray> *rays) {
        if (not DOF) {
            Ray result = Ray(pos, Point::minusPoint(pixel, pos), clip, std::numeric_limits<float>::infinity());
            rays->push_back(result);
        } else {
            Ray results;
            Point rp = randomPoint();
            for (int i = 0; i < 16; i++) {
                results = Ray(rp, Point::minusPoint(pixel, rp), clip, std::numeric_limits<float>::infinity());
                rays->push_back(results);
            }
        }
    }

    // Use sqrt(r) so uniformly sample on a disk.
    Point randomPoint() {
        float randomRroot = sqrt(randomRadius());
        float randomAng = randomAngle();
        float x = pos.x + randomRroot*cos(randomAng)*upD.x + randomRroot*sin(randomAng)*diskD.x;
        float y = pos.y + randomRroot*cos(randomAng)*upD.y + randomRroot*sin(randomAng)*diskD.y;
        float z = pos.z + randomRroot*cos(randomAng)*upD.z + randomRroot*sin(randomAng)*diskD.z;
        return Point(x, y, z);
    }

    float randomAngle() {
        float a = 2.0*PI*(float)rand()/((float)RAND_MAX);
        return a;
    }

    float randomRadius() {
        float r = 0.1*(float)rand()/((float)RAND_MAX);
        return r;  
    }
};

class Raytracer {
  public:
    Primitive* primitives;
    std::vector<Light*> *lights;
    Color *globalAm;
    float (*attenuations)[3];
    PhotonMap* pm;
    PhotonMap* cm;
    bool di;
    bool ci;
    bool ii;

    Raytracer() {
        if (debug) cout << "Zero-parameter constructor of Raytracer is called." << endl;
    }

    Raytracer(Primitive* p, std::vector<Light*> *l, Color* am) {
        primitives = p; lights = l; globalAm = am; di = false; ci = false; ii = false;
    }

    //emit photon for each light source
    void emitLightPhoton(){
        float lightPower = 6000.0f;
        float total = 200000;
        pm = new PhotonMap(total);
        cm = new PhotonMap(0.1*total);
        WPhoton p;
        float aver = total/(lights->size());
        for (std::vector<Light*>::size_type i = 0; i != lights->size(); i++) {
            #pragma omp parallel
            {
                while(pm->get_stored_photons() < aver){
                    p = lights->at(i)->emitPhoton();
                    firstPhotonTrace(p, 5);
                }
            }
        }
        pm->scale_photon_power(lightPower/total);
        pm->balance();
        cm->scale_photon_power(lightPower/total);
        cm->balance();

    }

    //first trace of photon that determines whether the photon should be stored in caustic map
    void firstPhotonTrace(WPhoton& p, int depth){
        if (depth <= 0) {
            return;
        }
        float tHit;
        Intersection in;
        BRDF brdf;        

        Ray ray = Ray(p.pos, p.dir, 0.00001, std::numeric_limits<float>::infinity());
        if (!primitives->intersect(ray, &tHit, &in)) {
            return;
        }


        in.primitive->getBRDF(in.lg, &brdf);
        if (Vector::dotMul(p.dir, in.lg.normal) > 0 && !(brdf.kt>0)){
            in.lg.normal = Vector::negate(in.lg.normal);
        }
        //refraction
        if (brdf.kt > 0) {
            LocalGeo local = in.lg;
            Vector rayDir = Vector::normalizeV(ray.dir);
            float n;
            float cosI = -1*Vector::dotMul(rayDir, local.normal);
            if (cosI > 0){
                n = AIR/brdf.rf;
                float cosT = sqrt(1 - sqr(n)*(1-sqr(cosI)));
                if (cosT >= 0.0){
                    Vector refractDir = Vector::addVector(Vector::multiVS(rayDir, n), Vector::multiVS(local.normal, (n*cosI - cosT)));
                    WPhoton np = WPhoton();
                    np.color = Color::multiplyC(p.color, brdf.kt);
                    np.pos = in.lg.pos;
                    np.dir = refractDir;
                    photonTrace(np, depth, true);
                } 
            }else{
                n = brdf.rf/AIR;
                float cosT = sqrt(1 - sqr(n)*(1-sqr(cosI)));
                if (cosT >= 0.0){
                    Vector refractDir = Vector::addVector(Vector::multiVS(rayDir, n), Vector::multiVS(Vector::negate(local.normal), (-n*cosI - cosT)));
                    WPhoton np = WPhoton();
                    np.color = Color::multiplyC(p.color, brdf.kt);
                    np.pos = in.lg.pos;
                    np.dir = refractDir;
                    photonTrace(np, depth, true);
                }

            }
        }else{
            float pr = max(max(brdf.kd.r + brdf.kr.r, brdf.kd.g + brdf.kr.g), brdf.kd.b + brdf.kr.b);
            float pd = pr*(brdf.kd.r + brdf.kd.g + brdf.kd.b)/(brdf.kd.r + brdf.kd.g + brdf.kd.b + brdf.kr.r + brdf.kr.g + brdf.kr.b);
            float ran = ((float)rand()/(RAND_MAX));
            //diffuse
            if (ran < pd){
                float Power[3]; Power[0] = p.color.r; Power[1] = p.color.g; Power[2] = p.color.b;
                float Pos[3]; Pos[0] = in.lg.pos.x; Pos[1] = in.lg.pos.y; Pos[2] = in.lg.pos.z;
                float Dir[3]; Dir[0] = p.dir.x; Dir[1] = p.dir.y; Dir[2] = p.dir.z;
                #pragma omp critical
                {
                pm->store(Power, Pos, Dir);
                }

                Vector nextDir;
                do{
                   nextDir.setValues(((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5);
                   nextDir.normalize();  
                } while(((float)rand()/(RAND_MAX)) >= Vector::dotMul(nextDir, in.lg.normal));
                WPhoton np = WPhoton(in.lg.pos, nextDir, Color::multiplyS(Color::multiplyC(p.color, brdf.kd), 1.0/pd));
                photonTrace(np, depth-1, false);



            //specular
            }else if (ran < pr){
                WPhoton np = WPhoton();
                np.color = Color::multiplyS(Color::multiplyC(p.color, brdf.kr), 1.0/(pr-pd));
                np.pos = in.lg.pos;
                Ray reflectRay = createReflectRay(in.lg, ray);
                np.dir = reflectRay.dir;
                photonTrace(np, depth-1, false);
            }else {
                return;
            }
        }
    }

    void photonTrace(WPhoton& p, int depth, bool caustic){
        if (depth <= 0) {
            return;
        }
        float tHit;
        Intersection in;
        BRDF brdf;        

        Ray ray = Ray(p.pos, p.dir, 0.00001, std::numeric_limits<float>::infinity());
        if (!primitives->intersect(ray, &tHit, &in)) {
            return;
        }

        in.primitive->getBRDF(in.lg, &brdf);
        if (Vector::dotMul(p.dir, in.lg.normal) > 0 && !(brdf.kt>0)){
            in.lg.normal = Vector::negate(in.lg.normal);
        }


        if (brdf.kt > 0) {
            LocalGeo local = in.lg;
            Vector rayDir = Vector::normalizeV(ray.dir);
            float n;
            float cosI = -1*Vector::dotMul(rayDir, local.normal);
            if (cosI > 0){
                n = AIR/brdf.rf;
                float cosT = sqrt(1 - sqr(n)*(1-sqr(cosI)));
                if (cosT >= 0.0){
                    Vector refractDir = Vector::addVector(Vector::multiVS(rayDir, n), Vector::multiVS(local.normal, (n*cosI - cosT)));
                    WPhoton np = WPhoton();
                    np.color = Color::multiplyC(p.color, brdf.kt);
                    np.pos = in.lg.pos;
                    np.dir = refractDir;
                    photonTrace(np, depth, caustic);
                } 
            }else{
                n = brdf.rf/AIR;
                float cosT = sqrt(1 - sqr(n)*(1-sqr(cosI)));
                if (cosT >= 0.0){
                    Vector refractDir = Vector::addVector(Vector::multiVS(rayDir, n), Vector::multiVS(Vector::negate(local.normal), (-n*cosI - cosT)));
                    WPhoton np = WPhoton();
                    np.color = Color::multiplyC(p.color, brdf.kt);
                    np.pos = in.lg.pos;
                    np.dir = refractDir;
                    photonTrace(np, depth, caustic);
                }

            }
        }else{
            float pr = max(max(brdf.kd.r + brdf.kr.r, brdf.kd.g + brdf.kr.g), brdf.kd.b + brdf.kr.b);
            float pd = pr*(brdf.kd.r + brdf.kd.g + brdf.kd.b)/(brdf.kd.r + brdf.kd.g + brdf.kd.b + brdf.kr.r + brdf.kr.g + brdf.kr.b);
            float ran = ((float)rand()/(RAND_MAX));
            if (ran < pd){
                float Power[3]; Power[0] = p.color.r; Power[1] = p.color.g; Power[2] = p.color.b;
                float Pos[3]; Pos[0] = in.lg.pos.x; Pos[1] = in.lg.pos.y; Pos[2] = in.lg.pos.z;
                float Dir[3]; Dir[0] = p.dir.x; Dir[1] = p.dir.y; Dir[2] = p.dir.z;
                #pragma omp critical
                {
                pm->store(Power, Pos, Dir);
                }
                if (caustic){
                    cm->store(Power, Pos, Dir);
                }
                Vector nextDir;
                do{
                   nextDir.setValues(((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5);
                   nextDir.normalize();  
                } while(((float)rand()/(RAND_MAX)) >= Vector::dotMul(nextDir, in.lg.normal));
                WPhoton np = WPhoton(in.lg.pos, nextDir, Color::multiplyS(Color::multiplyC(p.color, brdf.kd), 1.0/pd));
                photonTrace(np, depth-1, false);
            }else if (ran < pr){
                WPhoton np = WPhoton();
                np.color = Color::multiplyS(Color::multiplyC(p.color, brdf.kr), 1.0/(pr-pd));
                np.pos = in.lg.pos;
                Ray reflectRay = createReflectRay(in.lg, ray);
                np.dir = reflectRay.dir;
                photonTrace(np, depth-1, false);
            }else {
                return;
            }
        }
    }

    //find the irradiance estimate of the surface hit by the ray
    Color irradianceTrace(Ray& ray, bool first){
        float tH;
        Intersection is;
        BRDF brdf;

        if (primitives->intersect(ray, &tH, &is)) {
            is.primitive->getBRDF(is.lg, &brdf);

            if (Vector::dotMul(ray.dir, is.lg.normal) > 0 && !(brdf.kt>0)){
                is.lg.normal = Vector::negate(is.lg.normal);
            }

            if (tH<0.1&&first){


                Vector tDir = Vector();
                Ray r;

                Color irColor = Color();
                int numFinalGather = 50;
                float totalWeight = 0;
                float weight = 0;
                for(int z=0; z<numFinalGather; z++){
                    do {
                        tDir.setValues(((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5);
                        tDir.normalize();
                    } while(0 >= Vector::dotMul(is.lg.normal, tDir));
                    r = Ray(is.lg.pos, tDir, 0.01, std::numeric_limits<float>::infinity());
                    weight = Vector::dotMul(is.lg.normal, tDir);
                    irColor = Color::add(irColor, Color::multiplyS(irradianceTrace(r, false), weight));
                    totalWeight += weight;
                }
                return  Color::multiplyC(brdf.kd, Color::multiplyS(irColor, 1.0/totalWeight));

            }else {


                if (brdf.kr>0.9) {
                    Ray reflectRay = createReflectRay(is.lg, ray);
                    return irradianceTrace(reflectRay, first);

                } else if (brdf.kt > 0) {
                    LocalGeo local = is.lg;
                    Vector rayDir = Vector::normalizeV(ray.dir);

                    float n;
                    float cosI = -1*Vector::dotMul(rayDir, local.normal);
                    if (cosI > 0){
                        n = AIR/brdf.rf;
                        float cosT = sqrt(1 - sqr(n)*(1-sqr(cosI)));
                        if (cosT >= 0.0){
                            Vector refractDir = Vector::addVector(Vector::multiVS(rayDir, n), Vector::multiVS(local.normal, (n*cosI - cosT)));
                            Ray refractRay = Ray(local.pos, refractDir, 0.00001, std::numeric_limits<float>::infinity());
                            return irradianceTrace(refractRay, first);
                        } 
                    }else{
                        n = brdf.rf/AIR;
                        float cosT = sqrt(1 - sqr(n)*(1-sqr(cosI)));
                        if (cosT >= 0.0){
                            Vector refractDir = Vector::addVector(Vector::multiVS(rayDir, n), Vector::multiVS(Vector::negate(local.normal), (-n*cosI - cosT)));
                            Ray refractRay = Ray(local.pos, refractDir, 0.00001, std::numeric_limits<float>::infinity());
                            return irradianceTrace(refractRay, first);
                        }
                    }

                } else {
                    float irad[3];
                    float pos[3];
                    float normal[3];
                    pos[0] = is.lg.pos.x; pos[1] = is.lg.pos.y; pos[2] = is.lg.pos.z;
                    normal[0] = is.lg.normal.x; normal[1] = is.lg.normal.y; normal[2] = is.lg.normal.z;
                    pm->irradiance_estimate(irad, pos, normal, 0.5, 300);
                    return Color::multiplyC(Color(irad[0], irad[1],  irad[2]), brdf.kd);
                }
            }


        } else{
            return Color();
        }
    }

    void trace(Ray& ray, int depth, Color* color) {

        if (debug) cout << "trace is called with depth: " << depth << endl;

        float tHit;
        Intersection in;
        BRDF brdf;

        if (depth < 0) {
            color->setValues(0.0, 0.0, 0.0);
            return;
        }

        if (!primitives->intersect(ray, &tHit, &in)) {
            if (debug) cout << "In trace, not intersect!" << endl;
            color->setValues(0.0, 0.0, 0.0);
            return;
        }

        // Calculate self color including self-emission.
        Color selfColor;
        in.primitive->getBRDF(in.lg, &brdf);

        if (Vector::dotMul(ray.dir, in.lg.normal) > 0 && !(brdf.kt>0)){
            in.lg.normal = Vector::negate(in.lg.normal);
        }

        selfColor = Color::add(*globalAm, brdf.em);

        if (debug) cout << "In trace, globalAm is: " << globalAm->r << " " << globalAm->g << " " << globalAm->b << endl;
        if (debug) cout << "In trace, selfColor before lights is: " << selfColor.r << " " << selfColor.g << " " << selfColor.b << endl;
        *color = brdf.em;
        // Calculate color from light.
        Ray lray;
        float irad[3];
        float pos[3];           // surface position
        float normal[3];        // surface normal at pos
        float max_dist;         // max distance to look for photons
        int nphotons;           // number of photons to use
        if (ci){
            pos[0] = in.lg.pos.x; pos[1] = in.lg.pos.y; pos[2] = in.lg.pos.z;
            normal[0] = in.lg.normal.x; normal[1] = in.lg.normal.y; normal[2] = in.lg.normal.z;
            max_dist=1;
            nphotons=200; 

            cm->irradiance_estimate(irad, pos, normal, max_dist, nphotons);
            *color = Color::add(*color, Color::multiplyC(Color(irad[0], irad[1],  irad[2]), brdf.kd));
        }

        if (ii){

            Vector tDir = Vector();
            Ray r;

            Color irColor = Color();
            int numFinalGather = 50;
            float totalWeight = 0;
            float weight = 0;
            Ray reflectR = createReflectRay(in.lg, ray);
            for(int z=0; z<numFinalGather; z++){
                do {
                    tDir.setValues(((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5);
                    tDir.normalize();
                } while(0 >= Vector::dotMul(in.lg.normal, tDir));
                r = Ray(in.lg.pos, tDir, 0.01, std::numeric_limits<float>::infinity());
                weight = Vector::dotMul(in.lg.normal, tDir);
                irColor = Color::add(irColor, Color::multiplyS(irradianceTrace(r, true), weight));
                totalWeight += weight;
            }
            *color = Color::add(*color, Color::multiplyC(brdf.kd, Color::multiplyS(irColor, 1.0/totalWeight)));
        }


        if (di){
            int shadowSamples = 50;
            Color lcolor;
            float dist;
            float falloff;
            for (std::vector<Light*>::size_type i = 0; i != lights->size(); i++) {
                if (lights->at(i)->type() == 2){
                    Color tempColor = Color();
                    int count = 0;
                    while(count < shadowSamples){
                        lights->at(i)->generateLightRay(in.lg, lray, lcolor);
                        if (!primitives->intersectP(lray)) {
                            dist = lray.dir.length();
                            // falloff = 1/((*attenuations)[0] + (*attenuations)[1]*dist + (*attenuations)[2]*sqr(dist));
                            tempColor = Color::add(tempColor, Color::multiplyS(shading(in.lg, brdf, lray, lcolor,\
                             ray.pos), 1));
                            // tempColor = Color::add(tempColor, shading(in.lg, brdf, lray, lcolor, ray.pos));

                        }
                        count++;
                    }
                    selfColor = Color::add(selfColor, Color::multiplyS(tempColor, 1.0/shadowSamples));
                }else{
                    lights->at(i)->generateLightRay(in.lg, lray, lcolor);
                    if (!primitives->intersectP(lray)) {
                        if (lights->at(i)->type() == 1) {
                            selfColor = Color::add(selfColor,  shading(in.lg, brdf, lray, lcolor, ray.pos));
                        } else {
                            Point lsource = lights->at(i)->getSource();
                            dist = Point::minusPoint(in.lg.pos, lsource).length();
                            Color shade = shading(in.lg, brdf, lray, lcolor, ray.pos);
                            falloff = 1/((*attenuations)[0] + (*attenuations)[1]*dist + (*attenuations)[2]*sqr(dist));
                            selfColor = Color::add(selfColor, Color::multiplyS(shade, falloff));
                        }
                    }
                }
            }
            *color = Color::add(*color, Color::multiplyC(selfColor, Color(1.0-brdf.kt.r, 1.0-brdf.kt.g, 1.0-brdf.kt.b)));
        }

        
        // // Handle mirror reflection.
        if (brdf.kr > 0) {
            if (brdf.glossy == 0){
                if (debug) cout << "Start handling mirror reflection." << endl;
                Color tempColor;
                Ray reflectRay = createReflectRay(in.lg, ray);
                // Make a recursive call to trace the reflected ray.
                if (debug) cout << "Reflected ray is created, point to: " << reflectRay.dir.x << " " << reflectRay.dir.y << " " << reflectRay.dir.z << endl;
                trace(reflectRay, depth-1, &tempColor);
                if (debug) cout << "Reflection color is: " << tempColor.r << " " << tempColor.g << " " << tempColor.b << endl;
                *color = Color::add(*color, Color::multiplyC(tempColor, brdf.kr));
            } else{
                int sam = 5;
                float di = (sam-1)/2.0;
                float sc = 2.0*brdf.glossy/sam;
                Ray reflectRay = createReflectRay(in.lg, ray);
                Vector vecX = Vector::multiVS(Vector::normalizeV(Vector::crossMul(ray.dir, reflectRay.dir)), sc*reflectRay.dir.length());
                Vector vecY = Vector::multiVS(Vector::normalizeV(Vector::crossMul(vecX, reflectRay.dir)), sc*reflectRay.dir.length());

                Vector v;
                Ray r;
                Color tempColor;
                Color stepColor;
                for (int x = 0; x < sam; x++){
                    for (int y = 0; y < sam; y++){

                        v = Vector::addVector(reflectRay.dir, Vector::multiVS(vecX, x-di));
                        v = Vector::addVector(v, Vector::multiVS(vecY, y-di));
                        r = Ray(in.lg.pos, v, 0.00001, std::numeric_limits<float>::infinity());
                        trace(r, depth-1, &stepColor);
                        tempColor = Color::add(tempColor, stepColor);
                    }
                }
                tempColor = Color::multiplyS(tempColor, 1.0/(sam*sam));
                *color = Color::add(*color, Color::multiplyC(tempColor, brdf.kr));
            }
        }

        // Handle refraction.
        if (brdf.kt > 0) {
            LocalGeo local = in.lg;
            Color refractColor;
            Vector rayDir = Vector::normalizeV(ray.dir);
            float n;
            float cosI = -1*Vector::dotMul(rayDir, local.normal);
            if (cosI > 0){
                n = AIR/brdf.rf;
                float cosT = sqrt(1 - sqr(n)*(1-sqr(cosI)));
                if (cosT >= 0.0){
                    Vector refractDir = Vector::addVector(Vector::multiVS(rayDir, n), Vector::multiVS(local.normal, (n*cosI - cosT)));
                    Ray refractRay = Ray(local.pos, refractDir, 0.00001, std::numeric_limits<float>::infinity());
                    trace(refractRay, depth-1, &refractColor);
                    *color = Color::add(*color, Color::multiplyC(refractColor, brdf.kt));
                } 
            }else{
                n = brdf.rf/AIR;
                float cosT = sqrt(1 - sqr(n)*(1-sqr(cosI)));
                if (cosT >= 0.0){
                    Vector refractDir = Vector::addVector(Vector::multiVS(rayDir, n), Vector::multiVS(Vector::negate(local.normal), (-n*cosI - cosT)));
                    Ray refractRay = Ray(local.pos, refractDir, 0.00001, std::numeric_limits<float>::infinity());
                    trace(refractRay, depth-1, &refractColor);
                    *color = Color::add(*color, Color::multiplyC(refractColor, brdf.kt));

                }

            }
        }
    }

    Color shading (LocalGeo& lg, BRDF& brdf, Ray& lray, Color& lcolor, Point& eye) {
        return Color::add(calDiffuse(brdf.kd, lcolor, lg.normal, lray.dir), 
                            calSpecular(brdf.ks, lcolor, brdf.shininess, lg, lray.dir, eye));
    }

    Color calDiffuse (Color& kd, Color& lcolor, Vector& normal, Vector& ldir) {
        float c = Vector::dotMul(Vector::normalizeV(ldir), normal);
        if (debug) cout << "In calDiffuse, c is: " << c << endl;
        if (c < 0){
            return Color();
        }
        else{
            return Color::multiplyS( Color::multiplyC(kd, lcolor), c );
        }
    }

    Color calSpecular (Color& ks, Color& lcolor, float& s, LocalGeo& lg, Vector& ldir, Point& eye) {
        Vector norLdir = Vector::normalizeV(ldir);
        Vector reflect = Vector::addVector( Vector::negate(norLdir),  Vector::multiVS( lg.normal , 2.0 * Vector::dotMul(norLdir, lg.normal) ) );
        Vector view = Vector::normalizeV(Point::minusPoint( eye, lg.pos ));
        float c = Vector::dotMul (reflect, view);
        if (debug) cout << "In calSpecular, c is: " << c << endl;
        if (c < 0){
            return Color();
        }
        else{
            return Color::multiplyS( Color::multiplyC (ks, lcolor), pow(c, s) );
        }
    }

    Ray createReflectRay (LocalGeo& local, Ray& originRay) {
        Vector norLdir = Vector::normalizeV(originRay.dir);
        Vector reflect = Vector::addVector( norLdir,  Vector::multiVS( local.normal , 2.0 * Vector::dotMul(Vector::negate(norLdir), local.normal) ) );
        return Ray(local.pos, reflect, 0.00001, std::numeric_limits<float>::infinity());
    }

    Ray createGlossyRay (LocalGeo& local, Ray& originRay, float gl) {
        Vector norLdir = Vector::normalizeV(originRay.dir);
        Vector reflect = Vector::addVector( norLdir,  Vector::multiVS( local.normal , 2.0 * Vector::dotMul(Vector::negate(norLdir), local.normal) ) );
        reflect = Vector::normalizeV(reflect);
        reflect = Vector::addVector(Vector::normalizeV(reflect), Vector::multiVS(Vector(((float)rand()/(RAND_MAX))-0.5,\
         ((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5), 2*gl));
        return Ray(local.pos, reflect, 0.00001, std::numeric_limits<float>::infinity());
    }
};

class Film {
  public:
    int widthP, heightP;
    std::string imageName;
    Color** map;
    
    Film() {
        if (debug) cout << "Zero-parameter constructor of Film is called!" << endl;
    }

    Film(int w, int h, std::string name) {
        widthP = w;
        heightP = h;
        map = new Color*[widthP];
        for (int i = 0; i < widthP; ++i){
            map[i] = new Color[heightP];
        }
        imageName = name;
    }

    void commit(Sample& sample, Color& color) {
        map[sample.x][sample.y].setValues(color.r, color.g, color.b);
        if (debug) {
            cout << "In commit, color: " << color.r << " " << color.g << " " << color.b << endl;
        }
    }

    void writeImage() {
        FreeImage_Initialise ();
        FIBITMAP* bitmap = FreeImage_Allocate(widthP, heightP, 24); 
        RGBQUAD color;
        if (!bitmap){
            cout << "Cannot allocate images!" << endl;
            exit(1);
        }

        //Draws a gradient from blue to green:
        for (int i=0; i<widthP; i++) {
            for (int j=0; j<heightP; j++) {
                #ifdef OSX
                (map[i][j].r <= 1.0) ? (color.rgbBlue = map[i][j].r*255) : (color.rgbBlue = 255);
                (map[i][j].g <= 1.0) ? (color.rgbGreen = map[i][j].g*255) : (color.rgbGreen = 255);
                (map[i][j].b <= 1.0) ? (color.rgbRed = map[i][j].b*255) : (color.rgbRed = 255);   
                #else
                (map[i][j].r <= 1.0) ? (color.rgbRed = map[i][j].r*255) : (color.rgbRed = 255);
                (map[i][j].g <= 1.0) ? (color.rgbGreen = map[i][j].g*255) : (color.rgbGreen = 255);
                (map[i][j].b <= 1.0) ? (color.rgbBlue = map[i][j].b*255) : (color.rgbBlue = 255);
                #endif
                FreeImage_SetPixelColor(bitmap,i,j,&color);
            } 
        }

        const char* name = imageName.c_str();
        if (FreeImage_Save(FIF_PNG, bitmap, name, 0)) {
            cout << "Image successfully saved!" << endl;
        }
        FreeImage_DeInitialise (); //Cleanup!
    }
};

class Scene {
  public:
    // Image information.
    Point LU, LD, RU, RD;
    float widthP, heightP;

    // Camera information.
    Camera camera;
    
    // Sampler.
    Sampler sampler;

    // Film.
    Film film;

    // Scene name.
    std::string sceneName;

    // Raytracer.
    Raytracer raytracer;
    int depth;

    int numofPoints;

    Scene(){
        if (debug) cout << "Zero-parameter constructor of Scene is called." << endl;
    }

    Scene(std::string name) {
        sceneName = name;
    }

    void render() {
        Sample sample;
        std::vector<Point> points;
        Color color;
        Color stepColor;
        std::vector<Ray> rays;
        
        int total = widthP * heightP;
        int percent = total/100;
        int cur = 0;
        int per = 0;

        // Build AABB tree.
        cout << "Start building AABB trees." << endl;
        raytracer.primitives->setAABB();
        cout << "AABB tree has been constructed. Start rendering now." << endl;
        if (raytracer.ci||raytracer.ii){
            raytracer.emitLightPhoton();
            cout << "pm: " << raytracer.pm->get_stored_photons() << "\n";
            cout << "cm: " << raytracer.cm->get_stored_photons() << "\n";
        }
        int numofRays;
        std::vector<Point>::size_type i;
        std::vector<Ray>::size_type j;
        int sn = sampler.sampleNum();
        cout << "Maximum thread number is " << omp_get_max_threads() << endl;

        // For some weird reason, adding this pragma will cause compile error on mac.
        #ifndef OSX
        #pragma omp parallel for private(i, j, stepColor, color, rays, numofRays, points, sample)
        #endif
        for (int x = 0; x < sn; x++){
            #pragma omp critical
            {
            sampler.getSample(&sample, &points);
            }
            for(i = 0; i != points.size(); i++) {
                camera.generateRay(points[i], &rays);
                for(j = 0; j != rays.size(); j++) {
                    raytracer.trace(rays[j], depth, &stepColor);
                    color = Color::add(color, stepColor);
                }

                numofRays = rays.size();

                rays.clear();
            }
            // Average all the samples.
            color = Color::multiplyS(color, 1.0/(numofPoints*numofRays));
            
            #pragma omp critical
            {
            film.commit(sample, color);
            }
            // Reset color to 0 for next sample and clear points buffer and ray buffer.
            color.setValues(0.0, 0.0, 0.0);
            points.clear();

            if (debug) cout << "~~~~~~~~~~~~~~~~~~~~~~~DONE~~~~~~~~~~~~~~~~~~~~~~" << endl;

            #pragma omp critical
            {
            cur++;
            if (cur > percent) {
                // cout << omp_get_thread_num() << endl;
                per++;
                cur = 0;
                cout << per << " percentage done......." << endl;
            }
            }
        }
        cout << "100 percentage done......." << endl;
        film.writeImage();
    }
};

void runScene(std::string file, std::vector<std::string> objFiles) {

    //store variables and set stuff at the end.
    Scene scene = Scene("output.png");
    scene.depth = 5;
    int width, height;

    float samplesPP = 1; // Default value of samples per pixel.

    float clip = 0.0; // Default clipping plane.
    float focalDist = 1.0; // Default focal distance.


    std::vector<Primitive*> p;
    Primitive* prims = new AggregatePrimitive(&p);
    std::vector<Light*> li;
    Color* globalAm = new Color();
    float att[3] = {1, 0, 0};
    scene.raytracer = Raytracer(prims, &li, globalAm);
    scene.raytracer.attenuations = &att;
    scene.raytracer.di = true;
    int vI = 0, vnI = 0;
    Vector *vertices, *vertexNorm;

    std::vector<Matrix*> matrixStack;
    Material* m = new Material();

    std::ifstream inpfile(file.c_str());
    if(!inpfile.is_open()) {
        std::cout << "Unable to open file" << std::endl;
        exit(0);
    } else {
        std::string line;

        while(inpfile.good()) {
            std::vector<std::string> splitline;
            std::string buf;

            std::getline(inpfile,line);
            std::stringstream ss(line);

            while (ss >> buf) {
                splitline.push_back(buf);
            }

            //Ignore blank lines
            if(splitline.size() == 0) {
                continue;
            }

            //Ignore comments
            if(splitline[0][0] == '#') {
                continue;
            }

            //Valid commands:
            //size width height
            //  must be first command of file, controls image size
            else if(!splitline[0].compare("size")) {
                width = atoi(splitline[1].c_str());
                height = atoi(splitline[2].c_str());
                scene.widthP = width; 
                scene.heightP = height;
            }

            //maxdepth depth
            //  max # of bounces for ray (default 5)
            else if(!splitline[0].compare("maxdepth")) {
                scene.depth = atoi(splitline[1].c_str());
            }

            //output filename
            //  output file to write image to 
            else if(!splitline[0].compare("output")) {
                scene.sceneName = splitline[1];
                scene.film.imageName = scene.sceneName;
            }

            //Set samples per pixel vale.
            else if(!splitline[0].compare("SPP")) {
                samplesPP = atof(splitline[1].c_str());
                scene.numofPoints = samplesPP;
                scene.sampler.SPP = samplesPP;
                scene.sampler.step = sqrt(scene.sampler.SPP);
                if (samplesPP != 1) cout << "Using DRT with samples per pixel of " << samplesPP << "." << endl;
            }

            //camera lookfromx lookfromy lookfromz lookatx lookaty lookatz upx upy upz fov
            //  speciﬁes the camera in the standard way, as in homework 2.
            else if(!splitline[0].compare("camera")) {
                Point eye = Point(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                Point lookat = Point(atof(splitline[4].c_str()), atof(splitline[5].c_str()), atof(splitline[6].c_str()));
                Vector upDir = Vector::normalizeV(Vector(atof(splitline[7].c_str()), atof(splitline[8].c_str()), atof(splitline[9].c_str())));

                Vector viewDir = Point::minusPoint(lookat, eye);
                if (splitline.size() > 11) focalDist = atof(splitline[11].c_str());
                if (splitline.size() > 12) clip = atof(splitline[12].c_str());

                bool dof = false;
                if (focalDist != 1.0) {
                    dof = true;
                    cout << "Initiating depth of field with focal distance " << focalDist << "." << endl;
                }

                if (clip != 0.0) {
                    cout << "Settign clipping plane to " << clip << "." << endl;
                }

                Point center = Point::addVector(eye, Vector::multiVS(Vector::normalizeV(viewDir), focalDist));
                float fov = atof(splitline[10].c_str());

                float vertL = focalDist * tan(toRadian(fov/2));
                float horiL = vertL*width/height;
                
                Vector hod = Vector::normalizeV(Vector::crossMul(viewDir, upDir));
                Vector hV = Vector::multiVS(hod, horiL);
                Vector upd = Vector::normalizeV(Vector::crossMul(hV, viewDir));
                Vector vV = Vector::multiVS(upd, vertL);
                
                scene.LU = Point::addVector(center, Vector::addVector(vV, Vector::negate(hV)));
                scene.LD = Point::addVector(center, Vector::addVector(Vector::negate(vV), Vector::negate(hV)));
                scene.RU = Point::addVector(center, Vector::addVector(vV, hV));
                scene.RD = Point::addVector(center, Vector::addVector(Vector::negate(vV), hV));
                
                //Camera(Point p, float c, Vector upd, Vector viewd, Vector diskd);

                scene.camera = Camera(eye, clip, upd, Vector::normalizeV(viewDir), hod, dof);
                scene.sampler = Sampler(scene.LU, scene.LD, scene.RU, scene.RD, scene.widthP, scene.heightP, samplesPP);
                scene.numofPoints = samplesPP;
                scene.film = Film(scene.widthP, scene.heightP, scene.sceneName);
            }

            //sphere x y z radius
            //  Deﬁnes a sphere with a given position and radius.
            else if(!splitline[0].compare("sphere")) {
                Matrix mat = Matrix(Matrix::IDENTITY);
                Vector ori = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                Shape* s = new Sphere(ori, atof(splitline[4].c_str()));
                for (std::vector<Matrix*>::size_type i = 0; i != matrixStack.size(); i++) {
                    mat = Matrix::multiply(mat, (*matrixStack[i]));
                }
                Primitive* sph = new GeometricPrimitive(Transformation(mat), Transformation(Matrix::invert(mat)), s, *m);
                p.push_back (sph);
            }

            else if(!splitline[0].compare("di")) {
                scene.raytracer.di = (strcmp (splitline[1].c_str(), "true") == 0);
            }
            else if(!splitline[0].compare("ci")) {
                scene.raytracer.ci = (strcmp (splitline[1].c_str(), "true") == 0);
            }
            else if(!splitline[0].compare("ii")) {
                scene.raytracer.ii = (strcmp (splitline[1].c_str(), "true") == 0);
            }

            //maxverts number
            //  Deﬁnes a maximum number of vertices for later triangle speciﬁcations. 
            //  It must be set before vertices are deﬁned.
            else if(!splitline[0].compare("maxverts")) {
                vertices = new Vector[atoi(splitline[1].c_str())];
            }

            //maxvertnorms number
            //  Deﬁnes a maximum number of vertices with normals for later speciﬁcations.
            //  It must be set before vertices with normals are deﬁned.
            else if(!splitline[0].compare("maxvertnorms")) {
                vertexNorm = new Vector[2*atoi(splitline[1].c_str())];
            }

            //vertex x y z
            //  Deﬁnes a vertex at the given location.
            //  The vertex is put into a pile, starting to be numbered at 0.
            else if(!splitline[0].compare("vertex")) {
                vertices[vI] = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                vI++;
            }

            //vertexnormal x y z nx ny nz
            //  Similar to the above, but deﬁne a surface normal with each vertex.
            //  The vertex and vertexnormal set of vertices are completely independent
            //  (as are maxverts and maxvertnorms).
            else if(!splitline[0].compare("vertexnormal")) {
                vertexNorm[vnI] = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                vertexNorm[vnI+1] = Vector(atof(splitline[4].c_str()), atof(splitline[5].c_str()), atof(splitline[6].c_str()));
                vnI += 2;
            }

            //tri v1 v2 v3
            //  Create a triangle out of the vertices involved (which have previously been speciﬁed with
            //  the vertex command). The vertices are assumed to be speciﬁed in counter-clockwise order. Your code
            //  should internally compute a face normal for this triangle.
            else if(!splitline[0].compare("tri")) {
                Matrix mat = Matrix(Matrix::IDENTITY);
                Shape* t = new Triangle(vertices[atoi(splitline[1].c_str())], vertices[atoi(splitline[2].c_str())], vertices[atoi(splitline[3].c_str())]);
                for (std::vector<Matrix*>::size_type i = 0; i != matrixStack.size(); i++) {
                    mat = Matrix::multiply(mat, (*matrixStack[i]));
                }
                Primitive* tri = new GeometricPrimitive(Transformation(mat), Transformation(Matrix::invert(mat)), t, *m);
                p.push_back (tri);
            }

            //trinormal v1 v2 v3
            //  Same as above but for vertices speciﬁed with normals.
            //  In this case, each vertex has an associated normal, 
            //  and when doing shading, you should interpolate the normals 
            //  for intermediate points on the triangle.
            else if(!splitline[0].compare("trinormal")) {
                Matrix mat = Matrix(Matrix::IDENTITY);
                int i1 = atoi(splitline[1].c_str()), i2 = atoi(splitline[2].c_str()), i3 = atoi(splitline[3].c_str());
                Shape* t = new NormalTriangle(vertexNorm[2*i1], vertexNorm[2*i2], vertexNorm[2*i3], vertexNorm[2*i1+1],\
                    vertexNorm[2*i2+1], vertexNorm[2*i3+1]);
                for (std::vector<Matrix*>::size_type i = 0; i != matrixStack.size(); i++) {
                    mat = Matrix::multiply(mat, (*matrixStack[i]));
                }
                Primitive* tri = new GeometricPrimitive(Transformation(mat), Transformation(Matrix::invert(mat)), t, *m);
                p.push_back (tri);
            }

            //translate x y z
            //  A translation 3-vector
            else if(!splitline[0].compare("translate")) {
                Matrix thisM = Matrix(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()), Matrix::TRANSLATION);
                *matrixStack.back() = Matrix::multiply(*matrixStack.back(), thisM);
            }

            //rotate x y z angle
            //  Rotate by angle (in degrees) about the given axis as in OpenGL.
            else if(!splitline[0].compare("rotate")) {
                Vector axis = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                axis.normalize();
                Matrix thisM = Matrix(axis.x, axis.y, axis.z, toRadian(atof(splitline[4].c_str())), Matrix::ROTATION);
                *matrixStack.back() = Matrix::multiply(*matrixStack.back(), thisM);
            }

            //scale x y z
            //  Scale by the corresponding amount in each axis (a non-uniform scaling).
            else if(!splitline[0].compare("scale")) {
                Matrix thisM = Matrix(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()), Matrix::SCALING);
                *matrixStack.back() = Matrix::multiply(*matrixStack.back(), thisM);
            }

            //pushTransform
            //  Push the current modeling transform on the stack as in OpenGL. 
            //  You might want to do pushTransform immediately after setting 
            //   the camera to preserve the “identity” transformation.
            else if(!splitline[0].compare("pushTransform")) {
                matrixStack.push_back(new Matrix(Matrix::IDENTITY));
            }

            //popTransform
            //  Pop the current transform from the stack as in OpenGL. 
            //  The sequence of popTransform and pushTransform can be used if 
            //  desired before every primitive to reset the transformation 
            //  (assuming the initial camera transformation is on the stack as 
            //  discussed above).
            else if(!splitline[0].compare("popTransform")) {
                matrixStack.pop_back();
            }

            //directional x y z r g b
            //  The direction to the light source, and the color, as in OpenGL.
            else if(!splitline[0].compare("directional")) {
                Vector vec = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                Color col = Color(atof(splitline[4].c_str()), atof(splitline[5].c_str()), atof(splitline[6].c_str()));
                Light* l = new DirectionalLight(vec, col);
                li.push_back(l);
            }

            //point x y z r g b
            //  The location of a point source and the color, as in OpenGL.
            else if(!splitline[0].compare("point")) {
                Point poi = Point(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                Color col = Color(atof(splitline[4].c_str()), atof(splitline[5].c_str()), atof(splitline[6].c_str()));
                Light* l = new PointLight(poi, col);
                li.push_back(l);
            }

            else if(!splitline[0].compare("square")) {
                Point poi1 = Point(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                Point poi2 = Point(atof(splitline[4].c_str()), atof(splitline[5].c_str()), atof(splitline[6].c_str()));
                Point poi3 = Point(atof(splitline[7].c_str()), atof(splitline[8].c_str()), atof(splitline[9].c_str()));
                Vector vec = Vector(atof(splitline[10].c_str()), atof(splitline[11].c_str()), atof(splitline[12].c_str()));
                Color col = Color(atof(splitline[13].c_str()), atof(splitline[14].c_str()), atof(splitline[15].c_str()));
                Light* l = new SquareLight(poi1, poi2, poi3, vec, col);
                li.push_back(l);
            }

            //attenuation const linear quadratic
            //  Sets the constant, linear and quadratic attenuations 
            //  (default 1,0,0) as in OpenGL.
            else if(!splitline[0].compare("attenuation")) {
                att[0] = atof(splitline[1].c_str());
                att[1] = atof(splitline[2].c_str());
                att[2] = atof(splitline[3].c_str());
            }

            //ambient r g b
            //  The global ambient color to be added for each object 
            //  (default is .2,.2,.2)
            else if(!splitline[0].compare("ambient")) {
                globalAm->setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
            }

            //diﬀuse r g b
            //  speciﬁes the diﬀuse color of the surface.
            else if(!splitline[0].compare("diffuse")) {
                m->constantBRDF.kd.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
            }

            //specular r g b 
            //  speciﬁes the specular color of the surface.
            else if(!splitline[0].compare("specular")) {
                m->constantBRDF.ks.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                m->constantBRDF.kr.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
            }

            //shininess s
            //  speciﬁes the shininess of the surface.
            else if(!splitline[0].compare("shininess")) {
                m->constantBRDF.shininess = atof(splitline[1].c_str());
            }
            else if(!splitline[0].compare("glossy")) {
                m->constantBRDF.glossy = atof(splitline[1].c_str());
            }

            //emission r g b
            //  gives the emissive color of the surface.
            else if(!splitline[0].compare("emission")) {
                m->constantBRDF.em.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
            } 
            else if(!splitline[0].compare("refraction")) {
                m->constantBRDF.rf = atof(splitline[1].c_str());
            } 

            else if(!splitline[0].compare("transparency")) {
                m->constantBRDF.kt.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
            }
            else {
                std::cerr << "Unknown command: " << splitline[0] << std::endl;
            }
        }

        inpfile.close();
    }


    //parse obj files

    std::vector<Matrix*> mStack;
    const char Seperators[] = "/";

    for (int i = 0; i < objFiles.size(); i++){
        vI = 1; vnI = 1;
        Vector *v = new Vector[10000], *vn = new Vector[10000];
        m = new Material();
        mStack.clear();


        file = objFiles[i];
        std::ifstream inpfilee(file.c_str());
        scene.sceneName = file.c_str() + std::string(".png");
        scene.film.imageName = scene.sceneName;

        if(!inpfilee.is_open()) {
            std::cout << "Unable to open file" << std::endl;
            exit(0);
        } else {
            std::string line;

            while(inpfilee.good()) {
                std::vector<std::string> splitline;
                std::string buf;

                std::getline(inpfilee,line);
                std::stringstream ss(line);

                while (ss >> buf) {
                    splitline.push_back(buf);
                }

                //Ignore blank lines
                if(splitline.size() == 0) {
                    continue;
                }

                //Ignore comments
                if(splitline[0][0] == '#') {
                    continue;
                }

                //sphere x y z radius
                //  Deﬁnes a sphere with a given position and radius.
                else if(!splitline[0].compare("sphere")) {
                    Matrix mat = Matrix(Matrix::IDENTITY);
                    Vector ori = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                    Shape* s = new Sphere(ori, atof(splitline[4].c_str()));
                    for (std::vector<Matrix*>::size_type i = 0; i != mStack.size(); i++) {
                        mat = Matrix::multiply(mat, (*mStack[i]));
                    }
                    Primitive* sph = new GeometricPrimitive(Transformation(mat), Transformation(Matrix::invert(mat)), s, *m);
                    p.push_back (sph);
                }

                else if(!splitline[0].compare("v")) {
                    v[vI] = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                    vI++;
                }

                else if(!splitline[0].compare("vn")) {
                    vn[vnI] = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                    vnI ++;
                }

                else if(!splitline[0].compare("f")) {

                    std::vector<std::string> vvn;
                    std::string word;
                    stringstream st;
                    Matrix mat = Matrix(Matrix::IDENTITY);
                    for (std::vector<Matrix*>::size_type i = 0; i != mStack.size(); i++) {
                        mat = Matrix::multiply(mat, (*mStack[i]));
                    }
                    if (splitline.size() == 4){
                        st << (splitline[1].c_str()) << "/" << (splitline[2].c_str()) << "/" << (splitline[3].c_str()) << "/";
                        while(getline(st, word, '/')){
                            if (word != ""){
                                vvn.push_back(word);
                            }
                        }

                        Shape* t = new NormalTriangle(v[atoi(vvn[0].c_str())], v[atoi(vvn[2].c_str())], v[atoi(vvn[4].c_str())],\
                            vn[atoi(vvn[1].c_str())], vn[atoi(vvn[3].c_str())], vn[atoi(vvn[5].c_str())]);
                        Primitive* tri = new GeometricPrimitive(Transformation(mat), Transformation(Matrix::invert(mat)), t, *m);
                        p.push_back (tri);
                    } else{
                        st << (splitline[1].c_str()) << "/" << (splitline[2].c_str()) << "/" << (splitline[3].c_str()) << "/" << (splitline[4].c_str()) << "/";
                        while(getline(st, word, '/')){
                            if (word != ""){
                                vvn.push_back(word);
                            }
                        }

                        Shape* t1 = new NormalTriangle(v[atoi(vvn[0].c_str())], v[atoi(vvn[2].c_str())], v[atoi(vvn[4].c_str())],\
                            vn[atoi(vvn[1].c_str())], vn[atoi(vvn[3].c_str())], vn[atoi(vvn[5].c_str())]);
                        Shape* t2 = new NormalTriangle(v[atoi(vvn[0].c_str())], v[atoi(vvn[4].c_str())], v[atoi(vvn[6].c_str())],\
                            vn[atoi(vvn[1].c_str())], vn[atoi(vvn[5].c_str())], vn[atoi(vvn[7].c_str())]);
                        Primitive* tri1 = new GeometricPrimitive(Transformation(mat), Transformation(Matrix::invert(mat)), t1, *m);
                        Primitive* tri2 = new GeometricPrimitive(Transformation(mat), Transformation(Matrix::invert(mat)), t2, *m);
                        p.push_back (tri1);
                        p.push_back (tri2);




                    }
                }

                //translate x y z
                //  A translation 3-vector
                else if(!splitline[0].compare("translate")) {
                    Matrix thisM = Matrix(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()), Matrix::TRANSLATION);
                    *mStack.back() = Matrix::multiply(*mStack.back(), thisM);
                }

                //rotate x y z angle
                //  Rotate by angle (in degrees) about the given axis as in OpenGL.
                else if(!splitline[0].compare("rotate")) {
                    Vector axis = Vector(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                    axis.normalize();
                    Matrix thisM = Matrix(axis.x, axis.y, axis.z, toRadian(atof(splitline[4].c_str())), Matrix::ROTATION);
                    *mStack.back() = Matrix::multiply(*mStack.back(), thisM);
                }

                //scale x y z
                //  Scale by the corresponding amount in each axis (a non-uniform scaling).
                else if(!splitline[0].compare("scale")) {
                    Matrix thisM = Matrix(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()), Matrix::SCALING);
                    *mStack.back() = Matrix::multiply(*mStack.back(), thisM);
                }

                //pushTransform
                //  Push the current modeling transform on the stack as in OpenGL. 
                //  You might want to do pushTransform immediately after setting 
                //   the camera to preserve the “identity” transformation.
                else if(!splitline[0].compare("pushTransform")) {
                    mStack.push_back(new Matrix(Matrix::IDENTITY));
                }

                //popTransform
                //  Pop the current transform from the stack as in OpenGL. 
                //  The sequence of popTransform and pushTransform can be used if 
                //  desired before every primitive to reset the transformation 
                //  (assuming the initial camera transformation is on the stack as 
                //  discussed above).
                else if(!splitline[0].compare("popTransform")) {
                    mStack.pop_back();
                }

                //diﬀuse r g b
                //  speciﬁes the diﬀuse color of the surface.
                else if(!splitline[0].compare("diffuse")) {
                    m->constantBRDF.kd.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                }

                //specular r g b 
                //  speciﬁes the specular color of the surface.
                else if(!splitline[0].compare("specular")) {
                    m->constantBRDF.ks.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                    m->constantBRDF.kr.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                }

                //shininess s
                //  speciﬁes the shininess of the surface.
                else if(!splitline[0].compare("shininess")) {
                    m->constantBRDF.shininess = atof(splitline[1].c_str());
                }
                else if(!splitline[0].compare("glossy")) {
                    m->constantBRDF.glossy = atof(splitline[1].c_str());
                }

                //emission r g b
                //  gives the emissive color of the surface.
                else if(!splitline[0].compare("emission")) {
                    m->constantBRDF.em.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                } 
                else if(!splitline[0].compare("refraction")) {
                    m->constantBRDF.rf = atof(splitline[1].c_str());
                } 

                else if(!splitline[0].compare("transparency")) {
                    m->constantBRDF.kt.setValues(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
                }
                else {
                    std::cerr << "Unknown command: " << splitline[0] << std::endl;
                }
            }

            inpfilee.close();
        }
    }

    // Initialize a random number generator.
    srand((unsigned)time(0));
    cout << "di: " << scene.raytracer.di << endl;
    cout << "ci: " << scene.raytracer.ci << endl;
    cout << "ii: " << scene.raytracer.ii << endl;
    scene.render();
}

int main(int argc, char *argv[]) {

    if (argc == 1) {
        cout << "No input file specified!" << endl;
        exit(0);
    }
    std::string file  = argv[1];

    std::vector<std::string> objFiles;
    if (argc >= 3){
        for (int i =2; i<argc; i++){
            objFiles.push_back(argv[i]);
        }
    }
    runScene(file, objFiles);

    return 0;
};
