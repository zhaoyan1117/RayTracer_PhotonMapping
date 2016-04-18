#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <cstdlib>
#include <time.h>
#include <limits>

#define PI 3.14159265  // Should be used from mathlib
const float AIR = 1.0;
inline float sqr(float x) { return x*x; }
inline float toRadian(float x) { return x*PI/180; }
inline float divide (float x, float y) {
    if (y != 0) {
        return x/y;
    }
    else {
        if (x < 0) {
            return -(std::numeric_limits<float>::infinity());
        } else {
            return std::numeric_limits<float>::infinity();
        }
    }
}
inline float minV(float x, float y) { return x > y ? y : x; }
inline float maxV(float x, float y) { return x > y ? x : y; }
inline float half(float x, float y) { return (x+y) / 2; }
using namespace std;

// Global dubug flag.
bool debug = false;
bool AABBdebug = false;

// Class declaration.
class Vector;
class Point;
class Ray;
class Matrix;
class Color;
class BRDF;
class Material;
class Sample;
class LocalGeo;
class Transformation;
class Shape;
class Sphere;
class Triangle;
class NormalTriangle;
class Intersection;
class Primitive;
class GeometricPrimitive;
class AggregatePrimitive;
class Light;
class PointLight;
class DirectionalLight;

// Class implementation.
class Vector {
  public:
    float x, y, z;

    //Construct a zero vector.
    Vector () { 
        x = 0.0; y = 0.0; z = 0.0; 
    }

    //Construct vector with values v1, v2, and v3.
    Vector (float v1, float v2, float v3) { 
        x = v1; y = v2; z = v3; 
    }

    void setValues (float v1, float v2, float v3) { x = v1; y = v2; z = v3; }
    
    float length () {return sqrt(sqr(x) + sqr(y) + sqr(z));}
    
    //Normalize this if this is not a zero vector.
    void normalize() {
        float mag = this->length();
        if (mag != 0.0) {
            x = x/mag; y = y/mag; z = z/mag;
        }
    }

    //Normalize an input vector v if v is not a zero vector.
    static Vector normalizeV (Vector v) {
        float mag = v.length();
        Vector result;
        if (mag != 0.0) {
            result.setValues(v.x/mag, v.y/mag, v.z/mag);
        }
        return result;
    }

    //Return the dot product of vector v and w : v * w
    static float dotMul (Vector v, Vector w) {
        return v.x * w.x + v.y * w.y + v.z * w.z;
    }

    //Return the cross product of vector v and w : v X w
    static Vector crossMul (Vector v, Vector w) {
        float rx, ry, rz;
        rx = v.y * w.z - w.y * v.z;
        ry = v.z * w.x - w.z * v.x;
        rz = v.x * w.y - w.x * v.y;
        return Vector(rx, ry, rz);
    }

    //Return the sum of vector v and w.
    static Vector addVector (Vector v, Vector w) {
        Vector result (v.x + w.x, v.y + w.y, v.z + w.z);
        return result;
    }

    //Multiply vector v and scalar w.
    static Vector multiVS (Vector v, float w) {
        Vector result (v.x * w, v.y * w, v.z * w);
        return result;
    }

    //Negate an input value vector v.
    static Vector negate (Vector v) {
        Vector result (-v.x, -v.y, -v.z);
        return result;
    }
};

class Point {
  public:
    float x, y, z;

    Point () {
    	x = 0.0; y = 0.0; z = 0.0;
    }

    Point (float v1, float v2, float v3) { 
        x = v1; y = v2; z = v3; 
    }

    Vector toVector () {
        return Vector(x, y, z);
    }

    static Point addVector (Point& p, Vector v) {
        Point result (p.x + v.x, p.y + v.y, p.z + v.z);
        return result;
    }

    static Vector minusPoint (Point& e, Point& s) {
        Vector result (e.x - s.x, e.y - s.y, e.z - s.z);
        return result;
    }
};

//Represents the ray ray(t) = pos + t*dir, where tMin <= t <= tMax.
class Ray {
  public:
    Point pos;
    Vector dir;
    Vector inv_dir;
    bool sign[3];    
    float tMin, tMax;

    Ray () {
        if (debug) cout << "Zero-parameter constructor of Ray is called." << endl;
    }

    Ray (Point p, Vector d, float min, float max) {
        pos = p; dir = d; tMin = min; tMax = max;
        inv_dir = Vector(divide(1, dir.x), divide(1, dir.y), divide(1, dir.z));
        sign[0] = (inv_dir.x < 0);
        sign[1] = (inv_dir.y < 0);
        sign[2] = (inv_dir.z < 0);
    }

    Point calculate(float t) {
        if (t < tMin or t > tMax) {
            cout << "Ray::calculate: t is not in range." << endl;
            exit(1);
        } else {
            return Point::addVector(pos, Vector::multiVS(dir, t));
        }
    }
};

class Matrix {
  public:
    static const int TRANSLATION = 0;
    static const int SCALING = 1;
    static const int ROTATION = 2;
    static const int ZERO = 3;
    static const int IDENTITY = 4;
    static const int MIX = 5;
    float mat[4][4];
    int type;

    //create a identity matrix,
    Matrix () {
        type = Matrix::ZERO;
        mat[0][0] = 0.0; mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = 0.0;
        mat[1][0] = 0.0; mat[1][1] = 0.0; mat[1][2] = 0.0; mat[1][3] = 0.0;
        mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = 0.0; mat[2][3] = 0.0;
        mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 0.0;
    }

    Matrix (int t) {
        type = t;
        if (type != Matrix::IDENTITY) {
            cout << "One parameter constructor can only be called to build IDENTITY matrix.\n";
            exit(1);
        }
        mat[0][0] = 1.0; mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = 0.0;
        mat[1][0] = 0.0; mat[1][1] = 1.0; mat[1][2] = 0.0; mat[1][3] = 0.0;
        mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = 1.0; mat[2][3] = 0.0;
        mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 1.0;        
    }

    Matrix (float x, float y, float z, int t) {
        type = t;
        if (type == Matrix::TRANSLATION) {
            mat[0][0] = 1.0; mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = x;
            mat[1][0] = 0.0; mat[1][1] = 1.0; mat[1][2] = 0.0; mat[1][3] = y;
            mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = 1.0; mat[2][3] = z;
            mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 1.0;
        } else if (type == Matrix::SCALING) {
            mat[0][0] = x  ; mat[0][1] = 0.0; mat[0][2] = 0.0; mat[0][3] = 0.0;
            mat[1][0] = 0.0; mat[1][1] = y  ; mat[1][2] = 0.0; mat[1][3] = 0.0;
            mat[2][0] = 0.0; mat[2][1] = 0.0; mat[2][2] = z  ; mat[2][3] = 0.0;
            mat[3][0] = 0.0; mat[3][1] = 0.0; mat[3][2] = 0.0; mat[3][3] = 1.0;
        } else {
            cout << "Four-parameter constructor can only be called to build TRANSLATION or SCALING matrix.\n";
            exit(1);
        }
    }
    
    Matrix (float x, float y, float z, float angle, int t) {
        type = t;
        if (type != Matrix::ROTATION) {
            cout << "Five-parameter constructor can only be called to build ROTATION matrix.\n";
            exit(1);
        }
        mat[0][0] = cos(angle)+sqr(x)*(1-cos(angle)); mat[0][1] =  x*y*(1-cos(angle))-z*sin(angle); mat[0][2] =  y*sin(angle)+x*z*(1-cos(angle)); mat[0][3] = 0.0;
        mat[1][0] =  z*sin(angle)+x*y*(1-cos(angle)); mat[1][1] = cos(angle)+sqr(y)*(1-cos(angle)); mat[1][2] = -x*sin(angle)+y*z*(1-cos(angle)); mat[1][3] = 0.0;
        mat[2][0] = -y*sin(angle)+x*z*(1-cos(angle)); mat[2][1] =  x*sin(angle)+y*z*(1-cos(angle)); mat[2][2] = cos(angle)+sqr(z)*(1-cos(angle)); mat[2][3] = 0.0;
        mat[3][0] =                              0.0; mat[3][1] =                              0.0; mat[3][2] =                              0.0; mat[3][3] = 1.0;        
    }

    static Matrix invert(Matrix m) {
        Matrix result;
        float det;
        result.type = m.type;

        det = m.mat[0][0]*m.mat[1][1]*m.mat[2][2]*m.mat[3][3] + m.mat[0][0]*m.mat[1][2]*m.mat[2][3]*m.mat[3][1] + m.mat[0][0]*m.mat[1][3]*m.mat[2][1]*m.mat[3][2]\
            + m.mat[0][1]*m.mat[1][0]*m.mat[2][3]*m.mat[3][2] + m.mat[0][1]*m.mat[1][2]*m.mat[2][0]*m.mat[3][3] + m.mat[0][1]*m.mat[1][3]*m.mat[2][2]*m.mat[3][0]\
            + m.mat[0][2]*m.mat[1][0]*m.mat[2][1]*m.mat[3][3] + m.mat[0][2]*m.mat[1][1]*m.mat[2][3]*m.mat[3][0] + m.mat[0][2]*m.mat[1][3]*m.mat[2][0]*m.mat[3][1]\
            + m.mat[0][3]*m.mat[1][0]*m.mat[2][2]*m.mat[3][1] + m.mat[0][3]*m.mat[1][1]*m.mat[2][0]*m.mat[3][2] + m.mat[0][3]*m.mat[1][2]*m.mat[2][1]*m.mat[3][0]\
            - m.mat[0][0]*m.mat[1][1]*m.mat[2][3]*m.mat[3][2] - m.mat[0][0]*m.mat[1][2]*m.mat[2][1]*m.mat[3][3] - m.mat[0][0]*m.mat[1][3]*m.mat[2][2]*m.mat[3][1]\
            - m.mat[0][1]*m.mat[1][0]*m.mat[2][2]*m.mat[3][3] - m.mat[0][1]*m.mat[1][2]*m.mat[2][3]*m.mat[3][0] - m.mat[0][1]*m.mat[1][3]*m.mat[2][0]*m.mat[3][2]\
            - m.mat[0][2]*m.mat[1][0]*m.mat[2][3]*m.mat[3][1] - m.mat[0][2]*m.mat[1][1]*m.mat[2][0]*m.mat[3][3] - m.mat[0][2]*m.mat[1][3]*m.mat[2][1]*m.mat[3][0]\
            - m.mat[0][3]*m.mat[1][0]*m.mat[2][1]*m.mat[3][2] - m.mat[0][3]*m.mat[1][1]*m.mat[2][2]*m.mat[3][0] - m.mat[0][3]*m.mat[1][2]*m.mat[2][0]*m.mat[3][1];
        
        if (det == 0) 
        {
            if (debug) cout << "Matrix cannot be inverted since det is 0, returning a zero matrix." <<endl;
            return Matrix();
        } 

        result.mat[0][0] = m.mat[1][1]*m.mat[2][2]*m.mat[3][3] + m.mat[1][2]*m.mat[2][3]*m.mat[3][1] + m.mat[1][3]*m.mat[2][1]*m.mat[3][2]\
                  - m.mat[1][1]*m.mat[2][3]*m.mat[3][2] - m.mat[1][2]*m.mat[2][1]*m.mat[3][3] - m.mat[1][3]*m.mat[2][2]*m.mat[3][1];
        result.mat[0][0] = result.mat[0][0]/det;

        result.mat[0][1] = m.mat[0][1]*m.mat[2][3]*m.mat[3][2] + m.mat[0][2]*m.mat[2][1]*m.mat[3][3] + m.mat[0][3]*m.mat[2][2]*m.mat[3][1]\
                  - m.mat[0][1]*m.mat[2][2]*m.mat[3][3] - m.mat[0][2]*m.mat[2][3]*m.mat[3][1] - m.mat[0][3]*m.mat[2][1]*m.mat[3][2];
        result.mat[0][1] = result.mat[0][1]/det;

        result.mat[0][2] = m.mat[0][1]*m.mat[1][2]*m.mat[3][3] + m.mat[0][2]*m.mat[1][3]*m.mat[3][1] + m.mat[0][3]*m.mat[1][1]*m.mat[3][2]\
                  - m.mat[0][1]*m.mat[1][3]*m.mat[3][2] - m.mat[0][2]*m.mat[1][1]*m.mat[3][3] - m.mat[0][3]*m.mat[1][2]*m.mat[3][1];
        result.mat[0][2] = result.mat[0][2]/det;

        result.mat[0][3] = m.mat[0][1]*m.mat[1][3]*m.mat[2][2] + m.mat[0][2]*m.mat[1][1]*m.mat[2][3] + m.mat[0][3]*m.mat[1][2]*m.mat[2][1]\
                  - m.mat[0][1]*m.mat[1][2]*m.mat[2][3] - m.mat[0][2]*m.mat[1][3]*m.mat[2][1] - m.mat[0][3]*m.mat[1][1]*m.mat[2][2];
        result.mat[0][3] = result.mat[0][3]/det;

        result.mat[1][0] = m.mat[1][0]*m.mat[2][3]*m.mat[3][2] + m.mat[1][2]*m.mat[2][0]*m.mat[3][3] + m.mat[1][3]*m.mat[2][2]*m.mat[3][0]\
                  - m.mat[1][0]*m.mat[2][2]*m.mat[3][3] - m.mat[1][2]*m.mat[2][3]*m.mat[3][0] - m.mat[1][3]*m.mat[2][0]*m.mat[3][2];
        result.mat[1][0] = result.mat[1][0]/det;

        result.mat[1][1] = m.mat[0][0]*m.mat[2][2]*m.mat[3][3] + m.mat[0][2]*m.mat[2][3]*m.mat[3][0] + m.mat[0][3]*m.mat[2][0]*m.mat[3][2]\
                  - m.mat[0][0]*m.mat[2][3]*m.mat[3][2] - m.mat[0][2]*m.mat[2][0]*m.mat[3][3] - m.mat[0][3]*m.mat[2][2]*m.mat[3][0];
        result.mat[1][1] = result.mat[1][1]/det;

        result.mat[1][2] = m.mat[0][0]*m.mat[1][3]*m.mat[3][2] + m.mat[0][2]*m.mat[1][0]*m.mat[3][3] + m.mat[0][3]*m.mat[1][2]*m.mat[3][0]\
                  - m.mat[0][0]*m.mat[1][2]*m.mat[3][3] - m.mat[0][2]*m.mat[1][3]*m.mat[3][0] - m.mat[0][3]*m.mat[1][0]*m.mat[3][2];
        result.mat[1][2] = result.mat[1][2]/det;

        result.mat[1][3] = m.mat[0][0]*m.mat[1][2]*m.mat[2][3] + m.mat[0][2]*m.mat[1][3]*m.mat[2][0] + m.mat[0][3]*m.mat[1][0]*m.mat[2][2]\
                  - m.mat[0][0]*m.mat[1][3]*m.mat[2][2] - m.mat[0][2]*m.mat[1][0]*m.mat[2][3] - m.mat[0][3]*m.mat[1][2]*m.mat[2][0];
        result.mat[1][3] = result.mat[1][3]/det;

        result.mat[2][0] = m.mat[1][0]*m.mat[2][1]*m.mat[3][3] + m.mat[1][1]*m.mat[2][3]*m.mat[3][0] + m.mat[1][3]*m.mat[2][0]*m.mat[3][1]\
                  - m.mat[1][0]*m.mat[2][3]*m.mat[3][1] - m.mat[1][1]*m.mat[2][0]*m.mat[3][3] - m.mat[1][3]*m.mat[2][1]*m.mat[3][0];
        result.mat[2][0] = result.mat[2][0]/det;

        result.mat[2][1] = m.mat[0][0]*m.mat[2][3]*m.mat[3][1] + m.mat[0][1]*m.mat[2][0]*m.mat[3][3] + m.mat[0][3]*m.mat[2][1]*m.mat[3][0]\
                  - m.mat[0][0]*m.mat[2][1]*m.mat[3][3] - m.mat[0][1]*m.mat[2][3]*m.mat[3][0] - m.mat[0][3]*m.mat[2][0]*m.mat[3][1];
        result.mat[2][1] = result.mat[2][1]/det;

        result.mat[2][2] = m.mat[0][0]*m.mat[1][1]*m.mat[3][3] + m.mat[0][1]*m.mat[1][3]*m.mat[3][0] + m.mat[0][3]*m.mat[1][0]*m.mat[3][1]\
                  - m.mat[0][0]*m.mat[1][3]*m.mat[3][1] - m.mat[0][1]*m.mat[1][0]*m.mat[3][3] - m.mat[0][3]*m.mat[1][1]*m.mat[3][0];
        result.mat[2][2] = result.mat[2][2]/det;

        result.mat[2][3] = m.mat[0][0]*m.mat[1][3]*m.mat[2][1] + m.mat[0][1]*m.mat[1][0]*m.mat[2][3] + m.mat[0][3]*m.mat[1][1]*m.mat[2][0]\
                  - m.mat[0][0]*m.mat[1][1]*m.mat[2][3] - m.mat[0][1]*m.mat[1][3]*m.mat[2][0] - m.mat[0][3]*m.mat[1][0]*m.mat[2][1];
        result.mat[2][3] = result.mat[2][3]/det;

        result.mat[3][0] = m.mat[1][0]*m.mat[2][2]*m.mat[3][1] + m.mat[1][1]*m.mat[2][0]*m.mat[3][2] + m.mat[1][2]*m.mat[2][1]*m.mat[3][0]\
                  - m.mat[1][0]*m.mat[2][1]*m.mat[3][2] - m.mat[1][1]*m.mat[2][2]*m.mat[3][0] - m.mat[1][2]*m.mat[2][0]*m.mat[3][1];
        result.mat[3][0] = result.mat[3][0]/det;

        result.mat[3][1] = m.mat[0][0]*m.mat[2][1]*m.mat[3][2] + m.mat[0][1]*m.mat[2][2]*m.mat[3][0] + m.mat[0][2]*m.mat[2][0]*m.mat[3][1]\
                  - m.mat[0][0]*m.mat[2][2]*m.mat[3][1] - m.mat[0][1]*m.mat[2][1]*m.mat[3][2] - m.mat[0][2]*m.mat[2][1]*m.mat[3][0];
        result.mat[3][1] = result.mat[3][1]/det;

        result.mat[3][2] = m.mat[0][0]*m.mat[1][2]*m.mat[3][1] + m.mat[0][1]*m.mat[1][0]*m.mat[3][2] + m.mat[0][2]*m.mat[1][1]*m.mat[3][0]\
                  - m.mat[0][0]*m.mat[1][1]*m.mat[3][2] - m.mat[0][1]*m.mat[1][2]*m.mat[3][0] - m.mat[0][2]*m.mat[1][0]*m.mat[3][1];
        result.mat[3][2] = result.mat[3][2]/det;

        result.mat[3][3] = m.mat[0][0]*m.mat[1][1]*m.mat[2][2] + m.mat[0][1]*m.mat[1][2]*m.mat[2][0] + m.mat[0][2]*m.mat[1][0]*m.mat[2][1]\
                  - m.mat[0][0]*m.mat[1][2]*m.mat[2][1] - m.mat[0][1]*m.mat[1][0]*m.mat[2][2] - m.mat[0][2]*m.mat[1][1]*m.mat[2][0];
        result.mat[3][3] = result.mat[3][3]/det;

        return result;
    }
    
    static Matrix transpose(Matrix m) {
        Matrix result;
        result.type = m.type;
        for (int i=0; i<4; i++){
            for (int j=0; j<4; j++){
                result.mat[i][j] = m.mat[j][i];
            }
        }
        return result;
    }

    //return matrix multiplication of l times r.
    static Matrix multiply(Matrix l, Matrix r) {
        Matrix result;
        result.type = Matrix::MIX;

        for( int i = 0; i < 4; i++ ) {
            for( int j = 0; j < 4; j++ ) {
                for( int k = 0; k < 4; k++ ) {
                    result.mat[i][j] += l.mat[i][k] * r.mat[k][j];
                }
            }
        }

        return result;
    }
};

class Color {
  public:
    float r, g, b;

    Color() {
        r = 0.0; g = 0.0; b = 0.0;        
    }

    Color(float red, float green, float blue) {
        r = red; g = green; b = blue;
    }

    void setValues(float red, float green, float blue) {
        r = red, g = green, b = blue;
    }

    bool operator> (float f) {
        return sqrt(sqr(r) + sqr(g) + sqr(b)) > f;
    }

    static Color add(Color x, Color y) {
        Color result(x.r+y.r, x.g+y.g, x.b+y.b);
        return result;
    }

    static Color minus(Color x, Color y) {
        Color result(x.r-y.r, x.g-y.g, x.b-y.b);
        return result;
    }

    static Color multiplyC(Color n, Color m) {
        Color result(n.r*m.r, n.g*m.g, n.b*m.b);
        return result;
    }

    static Color multiplyS(Color x, float f) {
        Color result(f*x.r, f*x.g, f*x.b);
        return result;
    }
};

class BRDF {
  public:
    Color kd, ks, kr, em, kt;
    float rf, shininess;
    float glossy;

    BRDF() {
        kd = Color(); ks = Color(); kr = Color(); em = Color(); kt = Color(); rf = 1.0; shininess = 1.0; glossy = 0;
        if (debug) cout << "Zero-parameter constructor of BRDF is called." << endl;
    }

    BRDF(Color d, Color s, Color r, Color e, Color t, float f, float sh, float gl) {
        kd = d; ks = s; kr = r; em = e;
        kt = t; rf = f;
        shininess = sh;
        glossy = gl;
    }
};

class Material {
  public:
    BRDF constantBRDF;

    Material() {
        constantBRDF = BRDF();
        if (debug) cout << "Zero-parameter constructor of Material is called." << endl;
    }

    Material(BRDF brdf) {
        constantBRDF = brdf;
    }
    
    void getBRDF(LocalGeo& local, BRDF* brdf) {
        *brdf = constantBRDF;
    }
};

class Sample {
  public:
    int x, y;
    
    Sample() {
        x = 0; y = 0;
        if (debug) cout << "Zero-parameter constructor of Sample is called." << endl;

    }

    Sample(int xx, int yy) {
        x = xx; y = yy;
    }
};

class LocalGeo {
  public:
    Point pos;
    Vector normal;

    LocalGeo() {
        pos = Point(); normal = Vector();
        if (debug) cout << "Zero-parameter constructor of LocalGeo is called." << endl;
    }

    LocalGeo(Point p, Vector n) {
        pos = p, normal = n;
    }
};

class Transformation {
  public:
    Matrix m, minvt;

    Transformation() {
        m = Matrix(Matrix::IDENTITY); minvt = Matrix(Matrix::IDENTITY);
        if (debug) cout << "Zero-parameter constructor of Transformation is called." << endl;
    }

    Transformation(Matrix origin) {
        m = origin; minvt = Matrix::invert(Matrix::transpose(origin));
    }

    Point operator* (Point p) {
        float rx, ry, rz;
        rx = m.mat[0][0] * p.x + m.mat[0][1] * p.y + m.mat[0][2] * p.z + m.mat[0][3] * 1.0;
        ry = m.mat[1][0] * p.x + m.mat[1][1] * p.y + m.mat[1][2] * p.z + m.mat[1][3] * 1.0;
        rz = m.mat[2][0] * p.x + m.mat[2][1] * p.y + m.mat[2][2] * p.z + m.mat[2][3] * 1.0;
        return Point(rx, ry, rz);
    }

    Vector operator* (Vector v) {
        float rx, ry, rz;
        rx = m.mat[0][0] * v.x + m.mat[0][1] * v.y + m.mat[0][2] * v.z + m.mat[0][3] * 0.0;
        ry = m.mat[1][0] * v.x + m.mat[1][1] * v.y + m.mat[1][2] * v.z + m.mat[1][3] * 0.0;
        rz = m.mat[2][0] * v.x + m.mat[2][1] * v.y + m.mat[2][2] * v.z + m.mat[2][3] * 0.0;
        return Vector(rx, ry, rz);
    }

    Ray operator* (Ray r) {
        float px, py, pz, vx, vy, vz;
        px = m.mat[0][0] * r.pos.x + m.mat[0][1] * r.pos.y + m.mat[0][2] * r.pos.z + m.mat[0][3] * 1.0;
        py = m.mat[1][0] * r.pos.x + m.mat[1][1] * r.pos.y + m.mat[1][2] * r.pos.z + m.mat[1][3] * 1.0;
        pz = m.mat[2][0] * r.pos.x + m.mat[2][1] * r.pos.y + m.mat[2][2] * r.pos.z + m.mat[2][3] * 1.0;
        vx = m.mat[0][0] * r.dir.x + m.mat[0][1] * r.dir.y + m.mat[0][2] * r.dir.z + m.mat[0][3] * 0.0;
        vy = m.mat[1][0] * r.dir.x + m.mat[1][1] * r.dir.y + m.mat[1][2] * r.dir.z + m.mat[1][3] * 0.0;
        vz = m.mat[2][0] * r.dir.x + m.mat[2][1] * r.dir.y + m.mat[2][2] * r.dir.z + m.mat[2][3] * 0.0;
        return Ray(Point(px, py, pz), Vector(vx, vy, vz), r.tMin, r.tMax);
    }

    LocalGeo operator* (LocalGeo lg) {
        float px, py, pz, vx, vy, vz;
        px = m.mat[0][0] * lg.pos.x + m.mat[0][1] * lg.pos.y + m.mat[0][2] * lg.pos.z + m.mat[0][3] * 1.0;
        py = m.mat[1][0] * lg.pos.x + m.mat[1][1] * lg.pos.y + m.mat[1][2] * lg.pos.z + m.mat[1][3] * 1.0;
        pz = m.mat[2][0] * lg.pos.x + m.mat[2][1] * lg.pos.y + m.mat[2][2] * lg.pos.z + m.mat[2][3] * 1.0;
        vx = minvt.mat[0][0] * lg.normal.x + minvt.mat[0][1] * lg.normal.y + minvt.mat[0][2] * lg.normal.z + minvt.mat[0][3] * 0.0;
        vy = minvt.mat[1][0] * lg.normal.x + minvt.mat[1][1] * lg.normal.y + minvt.mat[1][2] * lg.normal.z + minvt.mat[1][3] * 0.0;
        vz = minvt.mat[2][0] * lg.normal.x + minvt.mat[2][1] * lg.normal.y + minvt.mat[2][2] * lg.normal.z + minvt.mat[2][3] * 0.0;
        return LocalGeo(Point(px, py, pz), Vector::normalizeV(Vector(vx, vy, vz)));
    }
};

class Shape {
  public:
    virtual bool intersect(Ray& ray, float* tHit, LocalGeo* local) = 0 ;
    virtual bool intersectP(Ray& ray) = 0;
    virtual void calAABBcorners(Point& v1, Point& v2, Point& v3, Point& v4, Point& v5, Point& v6, Point& v7, Point& v8) = 0;
    virtual void printShape() = 0;
};

class Sphere : public Shape {
  public:
    float radius;
    Vector center;
    
    Sphere () {
        radius = 1.0; center = Vector();
        if (debug) cout << "Zero-parameter constructor of Sphere is called." << endl;
    }

    Sphere (Vector c, float r) {
        radius = r; center = c;
    }

    bool intersect(Ray& ray, float* tHit, LocalGeo* local) {
        Vector eMinusC = Point::addVector(ray.pos, Vector::negate(center)).toVector();
        float B_2 =  sqr(Vector::dotMul(eMinusC, ray.dir));
        float AC_4 = Vector::dotMul(ray.dir, ray.dir) * ( Vector::dotMul(eMinusC, eMinusC) - sqr(radius) );
        float dis = B_2-AC_4;

        if (dis < 0) {
            if (debug) cout << "In sphere, dis < 0: " << dis << endl;
            return false;
        }

        float minusB = -Vector::dotMul(ray.dir, eMinusC);
        float D_2 = Vector::dotMul(ray.dir, ray.dir);

        float t;
        if (dis == 0) {
            t = minusB / D_2;
        } else if (dis > 0) {
            t = min(( minusB + sqrt(dis) ) / D_2, ( minusB - sqrt(dis) ) / D_2);
        }
        
        if (t < ray.tMin or t > ray.tMax) {
            return false;
        }

        if (debug) cout << "Hit in Sphere!" << endl;

        *tHit = t;

        local->pos = ray.calculate(*tHit);
        Vector n = Vector(local->pos.x - center.x, 
                                local->pos.y - center.y, 
                                local->pos.z - center.z);
        n.normalize();

        local->normal = n;

        if (debug) {
            cout << "In sphere, hit on: " << local->pos.x << " " << local->pos.y << " " << local->pos.z << endl;
            cout << "Normal is: " << local->normal.x << " " << local->normal.y << " " << local->normal.z << endl;  
        }
        return true;
    }

    bool intersectP(Ray& ray) {
        Vector eMinusC = Point::addVector(ray.pos, Vector::negate(center)).toVector();
        float B_2 =  sqr(Vector::dotMul(eMinusC, ray.dir));
        float AC_4 = Vector::dotMul(ray.dir, ray.dir) * ( Vector::dotMul(eMinusC, eMinusC) - sqr(radius) );
        float dis = B_2-AC_4;

        if (dis < 0) {
            return false;
        }

        float minusB = -Vector::dotMul(ray.dir, eMinusC);
        float D_2 = Vector::dotMul(ray.dir, ray.dir);

        float t;
        if (dis == 0) {
            t = minusB / D_2;
        } else if (dis > 0) {
            t = min(( minusB + sqrt(dis) ) / D_2, ( minusB - sqrt(dis) ) / D_2);
        }
        
        if (t < ray.tMin or t > ray.tMax) {
            return false;
        }

        return true;
    }

    void calAABBcorners(Point& v1, Point& v2, Point& v3, Point& v4, Point& v5, Point& v6, Point& v7, Point& v8) {
        // Min.
        v1.x = center.x - radius; v1.y = center.y - radius; v1.z = center.z - radius;

        v2.x = center.x - radius; v2.y = center.y + radius; v2.z = center.z - radius;

        v3.x = center.x - radius; v3.y = center.y - radius; v3.z = center.z + radius;

        v4.x = center.x - radius; v4.y = center.y + radius; v4.z = center.z + radius;

        v5.x = center.x + radius; v5.y = center.y - radius; v5.z = center.z - radius;

        v6.x = center.x + radius; v6.y = center.y + radius; v6.z = center.z - radius;

        v7.x = center.x + radius; v7.y = center.y - radius; v7.z = center.z + radius;
        // Max.
        v8.x = center.x + radius; v8.y = center.y + radius; v8.z = center.z + radius;
    }

    void printShape() {
        cout << "Sphere's center is: " << center.x << " " << center.y << " " << center.z << endl;
    }
};

class Triangle : public Shape {
  public:
    Vector A, B, C;

    Triangle() {
        A = Vector(); B = Vector(1.0, 0.0, 0.0); C = Vector(0.0, 1.0, 0.0);
        if (debug) cout << "Zero-parameter constructor of Triangle is called." << endl;
    }

    Triangle(Vector a, Vector b, Vector c) {
        A = a; B = b; C = c;
    }

    bool intersect(Ray& ray, float* tHit, LocalGeo* local) {
        Point E = ray.pos;
        Vector D = ray.dir;
        float a, b, c, d, e, f, g, h, i, j, k, l, M, t, gamma, beta;
        float tMin = ray.tMin, tMax = ray.tMax;

        a = A.x - B.x;
        b = A.y - B.y;
        c = A.z - B.z;
        d = A.x - C.x;
        e = A.y - C.y;
        f = A.z - C.z;
        g = D.x;
        h = D.y;
        i = D.z;
        j = A.x - E.x;
        k = A.y - E.y;
        l = A.z - E.z;
        M = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g);

        t = (f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c))/(-M);
        if (t < tMin or t > tMax) {
            return false;
        }
        gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c))/M;
        if (gamma < 0 or gamma > 1) {
            return false;
        }
        beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g))/M;
        if (beta < 0 or beta > (1-gamma)) {
            return false;
        }
        if (debug) cout << "In triangle hit!" << endl;
        *tHit = t;
        local->pos = ray.calculate(*tHit);

        Vector bMinusA = Vector::addVector(B, Vector::negate(A));
        Vector cMinusA = Vector::addVector(C, Vector::negate(A));
        local->normal =  Vector::normalizeV(Vector::crossMul(bMinusA, cMinusA)); //Orientation A-B-C counter-clockwise.
        return true;
    }

    bool intersectP(Ray& ray) {
        Point E = ray.pos;
        Vector D = ray.dir;
        float a, b, c, d, e, f, g, h, i, j, k, l, M, t, gamma, beta;
        float tMin = ray.tMin, tMax = ray.tMax;

        a = A.x - B.x;
        b = A.y - B.y;
        c = A.z - B.z;
        d = A.x - C.x;
        e = A.y - C.y;
        f = A.z - C.z;
        g = D.x;
        h = D.y;
        i = D.z;
        j = A.x - E.x;
        k = A.y - E.y;
        l = A.z - E.z;
        M = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g);

        t = (f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c))/(-M);
        if (t < tMin or t > tMax) {
            return false;
        }
        gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c))/M;
        if (gamma < 0 or gamma > 1) {
            return false;
        }
        beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g))/M;
        if (beta < 0 or beta > (1-gamma)) {
            return false;
        }
        if (debug) cout << "In triangle hit!" << endl;
        return true;
    }

    void calAABBcorners(Point& v1, Point& v2, Point& v3, Point& v4, Point& v5, Point& v6, Point& v7, Point& v8) {
        
        float minX = A.x, minY = A.y, minZ = A.z;
        float maxX = minX, maxY = minY, maxZ = minZ;

        // Check A.
        if (A.x < minX) minX = A.x;
        if (A.y < minY) minY = A.y;
        if (A.z < minZ) minZ = A.z;

        if (A.x > maxX) maxX = A.x;
        if (A.y > maxY) maxY = A.y;
        if (A.z > maxZ) maxZ = A.z;

        // Check B.
        if (B.x < minX) minX = B.x;
        if (B.y < minY) minY = B.y;
        if (B.z < minZ) minZ = B.z;

        if (B.x > maxX) maxX = B.x;
        if (B.y > maxY) maxY = B.y;
        if (B.z > maxZ) maxZ = B.z;

        // Check C.
        if (C.x < minX) minX = C.x;
        if (C.y < minY) minY = C.y;
        if (C.z < minZ) minZ = C.z;

        if (C.x > maxX) maxX = C.x;
        if (C.y > maxY) maxY = C.y;
        if (C.z > maxZ) maxZ = C.z;

        // Min.
        v1.x = minX; v1.y = minY; v1.z = minZ;

        v2.x = minX; v2.y = minY; v2.z = maxZ;

        v3.x = minX; v3.y = maxY; v3.z = minZ;

        v4.x = minX; v4.y = maxY; v4.z = maxZ;

        v5.x = maxX; v5.y = minY; v5.z = minZ;

        v6.x = maxX; v6.y = minY; v6.z = maxZ;

        v7.x = maxX; v7.y = maxY; v7.z = minZ;
        // Max.
        v8.x = maxX; v8.y = maxY; v8.z = maxZ;
    }

    void printShape() {
        cout << "Vertex A is: " << A.x << " " << A.y << " " << A.z << endl;
        cout << "Vertex B is: " << B.x << " " << B.y << " " << B.z << endl;
        cout << "Vertex C is: " << C.x << " " << C.y << " " << C.z << endl;
    }
};

class NormalTriangle : public Shape {
  public:
    Vector A, B, C;
    Vector nA, nB, nC;

    NormalTriangle() {
        A = Vector(); B = Vector(1.0, 0.0, 0.0); C = Vector(0.0, 1.0, 0.0);
        nA = Vector(0.0, 0.0, 1.0); nB = Vector(0.0, 0.0, 1.0); nC = Vector(0.0, 0.0, 1.0);
        if (debug) cout << "Zero-parameter constructor of NormalTriangle is called." << endl;
    }

    NormalTriangle(Vector a, Vector b, Vector c, Vector na, Vector nb, Vector nc) {
        A = a; B = b; C = c;
        nA = na; nB = nb; nC = nc;
        if (debug) {
            if (na.length() != 1.0 or nb.length() != 1.0 or nc.length() != 1.0) {
                cout << "Normal vectors must have length 1 in NormalTriangle!!" << endl;
                exit(1);
            }
        }
    }

    bool intersect(Ray& ray, float* tHit, LocalGeo* local) {
        Point E = ray.pos;
        Vector D = ray.dir;
        float a, b, c, d, e, f, g, h, i, j, k, l, M, t, gamma, beta;
        float tMin = ray.tMin, tMax = ray.tMax;

        a = A.x - B.x;
        b = A.y - B.y;
        c = A.z - B.z;
        d = A.x - C.x;
        e = A.y - C.y;
        f = A.z - C.z;
        g = D.x;
        h = D.y;
        i = D.z;
        j = A.x - E.x;
        k = A.y - E.y;
        l = A.z - E.z;
        M = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g);

        t = (f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c))/(-M);
        if (t < tMin or t > tMax) {
            return false;
        }
        gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c))/M;
        if (gamma < 0 or gamma > 1) {
            return false;
        }
        beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g))/M;
        if (beta < 0 or beta > (1-gamma)) {
            return false;
        }
        *tHit = t;
        local->pos = ray.calculate(*tHit);

        Vector bMinusA = Vector::addVector(B, Vector::negate(A));
        Vector cMinusA = Vector::addVector(C, Vector::negate(A));
        local->normal =  Vector::normalizeV(Vector::crossMul(bMinusA, cMinusA)); //Orientation A-B-C counter-clockwise.
        setNormal(local->normal, ray.calculate(t));
        return true;
    }

    bool intersectP(Ray& ray) {
        Point E = ray.pos;
        Vector D = ray.dir;
        float a, b, c, d, e, f, g, h, i, j, k, l, M, t, gamma, beta;
        float tMin = ray.tMin, tMax = ray.tMax;

        a = A.x - B.x;
        b = A.y - B.y;
        c = A.z - B.z;
        d = A.x - C.x;
        e = A.y - C.y;
        f = A.z - C.z;
        g = D.x;
        h = D.y;
        i = D.z;
        j = A.x - E.x;
        k = A.y - E.y;
        l = A.z - E.z;
        M = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g);

        t = (f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c))/(-M);
        if (t < tMin or t > tMax) {
            return false;
        }
        gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c))/M;
        if (gamma < 0 or gamma > 1) {
            return false;
        }
        beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g))/M;
        if (beta < 0 or beta > (1-gamma)) {
            return false;
        }
        return true;
    }

    void setNormal(Vector& localNormal, Point hitPoint) {
        Vector h = Vector::negate(hitPoint.toVector());
        Vector hA = Vector::addVector(A, h); 
        Vector hB = Vector::addVector(B, h); 
        Vector hC = Vector::addVector(C, h);
        float areaA = (Vector::crossMul(hB, hC)).length();
        float areaB = (Vector::crossMul(hC, hA)).length();
        float areaC = (Vector::crossMul(hA, hB)).length();
        float total = areaA + areaB + areaC;

        localNormal = Vector::addVector(Vector::addVector(Vector::multiVS(nA, areaA/total), Vector::multiVS(nB, areaB/total)), Vector::multiVS(nC, areaC/total));
        if (localNormal.length() != 1.0) localNormal.normalize();
    }

    void calAABBcorners(Point& v1, Point& v2, Point& v3, Point& v4, Point& v5, Point& v6, Point& v7, Point& v8) {
        
        float minX = A.x, minY = A.y, minZ = A.z;
        float maxX = minX, maxY = minY, maxZ = minZ;

        // Check A.
        if (A.x < minX) minX = A.x;
        if (A.y < minY) minY = A.y;
        if (A.z < minZ) minZ = A.z;

        if (A.x > maxX) maxX = A.x;
        if (A.y > maxY) maxY = A.y;
        if (A.z > maxZ) maxZ = A.z;

        // Check B.
        if (B.x < minX) minX = B.x;
        if (B.y < minY) minY = B.y;
        if (B.z < minZ) minZ = B.z;

        if (B.x > maxX) maxX = B.x;
        if (B.y > maxY) maxY = B.y;
        if (B.z > maxZ) maxZ = B.z;

        // Check C.
        if (C.x < minX) minX = C.x;
        if (C.y < minY) minY = C.y;
        if (C.z < minZ) minZ = C.z;

        if (C.x > maxX) maxX = C.x;
        if (C.y > maxY) maxY = C.y;
        if (C.z > maxZ) maxZ = C.z;

        // Min.
        v1.x = minX; v1.y = minY; v1.z = minZ;

        v2.x = minX; v2.y = minY; v2.z = maxZ;

        v3.x = minX; v3.y = maxY; v3.z = minZ;

        v4.x = minX; v4.y = maxY; v4.z = maxZ;

        v5.x = maxX; v5.y = minY; v5.z = minZ;

        v6.x = maxX; v6.y = minY; v6.z = maxZ;

        v7.x = maxX; v7.y = maxY; v7.z = minZ;
        // Max.
        v8.x = maxX; v8.y = maxY; v8.z = maxZ;
    }

    void printShape() {
        cout << "Vertex A is: " << A.x << " " << A.y << " " << A.z << endl;
        cout << "Vertex B is: " << B.x << " " << B.y << " " << B.z << endl;
        cout << "Vertex C is: " << C.x << " " << C.y << " " << C.z << endl;
    }
};

class Intersection {
  public:
    LocalGeo lg;
    Primitive* primitive;

    Intersection() {
        lg = LocalGeo(); primitive = NULL;
        if (debug) cout << "Zero-parameter constructor of Intersection is called." << endl;
    }

    Intersection(LocalGeo local, Primitive* p) {
        lg = local; primitive = p;
    }
};

class Primitive {
  public:
    Point min, max;
    bool Leaf;
    virtual bool intersect(Ray& ray, float* thit, Intersection* in) =0;
    virtual bool intersectP(Ray& ray) =0;
    virtual void getBRDF(LocalGeo& local, BRDF* brdf) =0;
    virtual void setAABB() =0;
    virtual void getAABB(Point& in, Point& ax) =0;
    virtual bool isLeaf() =0;
    virtual void setLeaf(bool l) =0;
    virtual bool intersectAABB(Ray& r, Point& in, Point& ax) = 0;
    virtual void printShape() =0;
};

class GeometricPrimitive : public Primitive {
  public:
    Transformation objToWorld, worldToObj;
    Shape* shape;
    Material mat;
    bool isSet;

    GeometricPrimitive() {
        if (debug) cout << "Zero-parameter constructor of GeometricPrimitive is called." << endl;
        isSet = false;
    }

    GeometricPrimitive(Transformation o2w, Transformation w2o, Shape* s, Material m) {
        objToWorld = o2w; worldToObj = w2o;
        shape = s; mat = m;
        isSet = false;
    }

    //bool intersectAABB(Ray& r, Point& in, Point& ax)
    bool intersect(Ray& ray, float* thit, Intersection* in)  {
        if (intersectAABB(ray, min, max)) {
            Ray oray;
            (worldToObj.m.type == Matrix::IDENTITY) ? (oray = ray) : (oray = worldToObj*ray);

            LocalGeo olocal;                                 
            if (!shape->intersect(oray, thit, &olocal)) return false;
            in->primitive = this;

            (objToWorld.m.type == Matrix::IDENTITY) ? (in->lg = olocal) : (in->lg = objToWorld*olocal);
            return true;
        } else {
            return false;
        }
    }

    bool intersectP(Ray& ray) {
        if (intersectAABB(ray, min, max)) {
            Ray oray;
            (worldToObj.m.type == Matrix::IDENTITY) ? (oray = ray) : (oray = worldToObj*ray);
            return shape->intersectP(oray);
        } else {
            return false;
        }
    }
    
    void getBRDF(LocalGeo& local, BRDF* brdf) {

        mat.getBRDF(local, brdf);
    }

    void setAABB() {
        if (not isSet) {

            Point v1, v2, v3, v4, v5, v6, v7, v8;
            
            shape->calAABBcorners(v1,v2,v3,v4,v5,v6,v7,v8);

            if (objToWorld.m.type != Matrix::IDENTITY) {
                v1 = objToWorld*v1; v2 = objToWorld*v2;
                v3 = objToWorld*v3; v4 = objToWorld*v4;
                v5 = objToWorld*v5; v6 = objToWorld*v6;
                v7 = objToWorld*v7; v8 = objToWorld*v8;
            }

            Point vertices[8] = {v1, v2, v3, v4, v5, v6, v7, v8};

            // Initialize with the first vertex.
            min.x = vertices[0].x; min.y = vertices[0].y; min.z = vertices[0].z;
            max.x = min.x; max.y = min.y; max.z = min.z;

            // Loop over all the other vertices to calculate min and max.
            for (int i = 1; i < 8; i++) {
                if (vertices[i].x < min.x) min.x = vertices[i].x;
                if (vertices[i].y < min.y) min.y = vertices[i].y;
                if (vertices[i].z < min.z) min.z = vertices[i].z;

                if (vertices[i].x > max.x) max.x = vertices[i].x;
                if (vertices[i].y > max.y) max.y = vertices[i].y;
                if (vertices[i].z > max.z) max.z = vertices[i].z;
            }
            isSet = true;       
        }
    }

    void getAABB(Point& in, Point& ax) {
        in = min; ax = max;
    }

    bool intersectAABB(Ray& r, Point& in, Point& ax) {
        Point bounds[2] = {in, ax};
        float tmin, tmax, tymin, tymax, tzmin, tzmax;

        tmin = (bounds[r.sign[0]].x - r.pos.x) * r.inv_dir.x;
        tmax = (bounds[1-r.sign[0]].x - r.pos.x) * r.inv_dir.x;
        tymin = (bounds[r.sign[1]].y - r.pos.y) * r.inv_dir.y;
        tymax = (bounds[1-r.sign[1]].y - r.pos.y) * r.inv_dir.y;
        
        if ( (tmin > tymax) || (tymin > tmax) ) return false;
        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;
        
        tzmin = (bounds[r.sign[2]].z - r.pos.z) * r.inv_dir.z;
        tzmax = (bounds[1-r.sign[2]].z - r.pos.z) * r.inv_dir.z;
        
        if ( (tmin > tzmax) || (tzmin > tmax) ) return false;
        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;
        return ( (tmin < r.tMax) && (tmax > r.tMin) );
    }

    bool isLeaf() {
        return true;
    }
    
    void setLeaf(bool l) {
        if (debug) cout << "GeometricPrimitive must be a leaf!" << endl;
    }

    void printShape() {
        shape->printShape();
    }
};

class AggregatePrimitive : public Primitive {
  public:
    std::vector<Primitive*> *list;
    Primitive* children[2];

    AggregatePrimitive() {
        if (debug) cout << "Zero-parameter constructor of AggregatePrimitive is called." << endl;
    }

    AggregatePrimitive(std::vector<Primitive*> *l) {
        list = l;
    }

    //bool intersectAABB(Ray& r, Point& in, Point& ax)
    bool intersect(Ray& ray, float* thit, Intersection* in) {

        if (intersectAABB(ray, min, max)) {

            *thit = std::numeric_limits<float>::infinity();
            float curT;
            Intersection curI;
            bool result = false;

            if (isLeaf()) {
                for(std::vector<Primitive*>::size_type i = 0; i != list->size(); i++) {
                    if ((*list)[i]->intersect(ray, &curT, &curI)) {
                        result = true;
                        if (curT < *thit) {
                            *thit = curT;
                            in->lg = curI.lg; in->primitive = curI.primitive;
                        }
                    }
                }
            } else {
                if (children[0]->intersect(ray, &curT, &curI)) {
                    result = true;
                    if (curT < *thit) {
                        *thit = curT;
                        in->lg = curI.lg; in->primitive = curI.primitive;
                    }
                }
                if (children[1]->intersect(ray, &curT, &curI)) {
                    result = true;
                    if (curT < *thit) {
                        *thit = curT;
                        in->lg = curI.lg; in->primitive = curI.primitive;
                    }
                }
            }
            return result;

        } else {
            return false;
        }
    }

    bool intersectP(Ray& ray) {

        if (intersectAABB(ray, min, max)) {
            bool result = false;
            if (Leaf) {
                for(std::vector<Primitive*>::size_type i = 0; i != list->size(); i++) {
                    if ((*list)[i]->intersectP(ray)) {
                        if (debug) cout << "Hit in aggre!" << endl;
                        result = true;
                    }
                }
            } else {
                if (children[0]->intersectP(ray)) {
                    result = true;
                }
                if (children[1]->intersectP(ray)) {
                    result = true;
                }                
            }
            return result;
        } else {
            return false;
        }

    }

    void getBRDF(LocalGeo& local, BRDF* brdf) {
        cout << "AggregatePrimitive's getBRDF should never be called.\n";
        exit(1);
    }

    void setAABB() {
        // Initial lize with the first primitives.
        Point in, ax;
        if (AABBdebug) (*list)[0]->printShape();
        (*list)[0]->setAABB();
        (*list)[0]->getAABB(in, ax);
        min.x = in.x; min.y = in.y; min.z = in.z;
        max.x = ax.x; max.y = ax.y; max.z = ax.z;
        // Loop over all the rest primitives to calculate min and max.

        for(std::vector<Primitive*>::size_type i = 1; i != list->size(); i++) {
            if (AABBdebug) (*list)[i]->printShape();
            (*list)[i]->setAABB();
            (*list)[i]->getAABB(in, ax);

            if (in.x < min.x) min.x = in.x;
            if (in.y < min.y) min.y = in.y;
            if (in.z < min.z) min.z = in.z;

            if (ax.x > max.x) max.x = ax.x;
            if (ax.y > max.y) max.y = ax.y;
            if (ax.z > max.z) max.z = ax.z;
        }

        if (AABBdebug) cout << "LEVEL DONE ______________________________________" << endl;
        // Partition.
        float longest = maxV( maxV(max.x - min.x, max.y - min.y), max.z - min.z);
        float mid;
        std::vector<Primitive*> *left = new std::vector<Primitive*>;
        std::vector<Primitive*> *right = new std::vector<Primitive*>;

        if (longest == (max.x - min.x)) {
            mid = half (max.x, min.x);
            for(std::vector<Primitive*>::size_type i = 0; i != list->size(); i++) {
                (*list)[i]->getAABB(in, ax);
                if (half(ax.x, in.x) < mid) {
                    left->push_back((*list)[i]);
                }
                else {
                    right->push_back((*list)[i]);
                }
            }
        } else if (longest == (max.y - min.y)) {
            mid = half (max.y, min.y);
            for(std::vector<Primitive*>::size_type i = 0; i != list->size(); i++) {
                (*list)[i]->getAABB(in, ax);
                if (half(ax.y, in.y) < mid) {
                    left->push_back((*list)[i]);
                }
                else {
                    right->push_back((*list)[i]);
                }
            }
        } else if (longest == (max.z - min.z)) {
            mid = half (max.z, min.z);
            for(std::vector<Primitive*>::size_type i = 0; i != list->size(); i++) {
                (*list)[i]->getAABB(in, ax);
                if (half(ax.z, in.z) < mid) {
                    left->push_back((*list)[i]);
                }
                else {
                    right->push_back((*list)[i]);
                }
            }
        }

        // Recursive call to build AABB tree and base case check.
        if ((left->size() == 0) or (right->size() == 0)) {
            setLeaf(true);
            if (AABBdebug) cout << "This is a leaf.~~~~~~~~~~~~~~~~~~" << endl;
        } else {
            if (AABBdebug) cout << "Building left children." << endl;
            if (left->size() == 1) {
                children[0] = (*left)[0];
                
                if (AABBdebug) children[0]->printShape();
                if (AABBdebug) cout << "Left leaf reached.~~~~~~~~~~~~~~~~~~" << endl;
            } else {
                children[0] = new AggregatePrimitive(left);
                children[0]->setAABB();
            }

            if (AABBdebug) cout << "Building right children." << endl;
            if (right->size() == 1) {
                children[1] = (*right)[0];

                if (AABBdebug) children[1]->printShape();
                if (AABBdebug) cout << "Right leaf reached.~~~~~~~~~~~~~~~~~~" << endl;
            } else {
                children[1] = new AggregatePrimitive(right);
                children[1]->setAABB();
            }
        }
    }

    void getAABB(Point& in, Point& ax) {
        in = min; ax = max;
    }

    bool intersectAABB(Ray& r, Point& in, Point& ax) {
        Point bounds[2] = {in, ax};
        float tmin, tmax, tymin, tymax, tzmin, tzmax;

        tmin = (bounds[r.sign[0]].x - r.pos.x) * r.inv_dir.x;
        tmax = (bounds[1-r.sign[0]].x - r.pos.x) * r.inv_dir.x;
        tymin = (bounds[r.sign[1]].y - r.pos.y) * r.inv_dir.y;
        tymax = (bounds[1-r.sign[1]].y - r.pos.y) * r.inv_dir.y;
        
        if ( (tmin > tymax) || (tymin > tmax) ) return false;
        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;
        
        tzmin = (bounds[r.sign[2]].z - r.pos.z) * r.inv_dir.z;
        tzmax = (bounds[1-r.sign[2]].z - r.pos.z) * r.inv_dir.z;
        
        if ( (tmin > tzmax) || (tzmin > tmax) ) return false;
        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;
        return ( (tmin < r.tMax) && (tmax > r.tMin) );
    }

    bool isLeaf() {
        return Leaf;
    }
    
    void setLeaf(bool l) {
        Leaf = l;
    }

    void printShape() {}
};

//Photon representation in world space
class WPhoton {
  public:
    Color color;
    Point pos;
    Vector dir;

    WPhoton() {
        pos = Point();
        dir = Vector();
        color = Color();
    }

    WPhoton(Point p, Vector d, Color c) {
        pos = p; dir = d; color = c;
    }

    void setValues(Point p, Vector d, Color c){
        pos = p; dir = d; color = c;
    }
};

class Light {
  public:
    virtual void generateLightRay(LocalGeo& local, Ray& lray, Color& lcolor) = 0;
    virtual int type() = 0;
    virtual Point getSource() = 0;
    virtual WPhoton emitPhoton() = 0;
};

class PointLight : public Light {
  public:
    Point source;
    Color color;
    
    PointLight() {
        if (debug) cout << "Zero-parameter constructor of PointLight is called." << endl;
        source = Point(); color = Color();
    }

    PointLight(Point s, Color c) {
        source = s; color = c;
    }

    void generateLightRay(LocalGeo& local, Ray& lray, Color& lcolor) {
        Vector dir = Point::minusPoint(source, local.pos);
        lray.pos = local.pos;
        lray.dir = dir;
        lray.inv_dir = Vector(divide(1, lray.dir.x), divide(1, lray.dir.y), divide(1, lray.dir.z));
        lray.sign[0] = (lray.inv_dir.x < 0);
        lray.sign[1] = (lray.inv_dir.y < 0);
        lray.sign[2] = (lray.inv_dir.z < 0);
        lray.tMin = 0.00001;
        lray.tMax = 1;
        lcolor = color;
    }

    //type 0 indicates point light
    int type() {
        return 0;
    }

    Point getSource() {
        return source;
    }

    //sample a photon
    WPhoton emitPhoton() {
        WPhoton p;
        Vector photonDir;
        photonDir.setValues(((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5);
        photonDir.normalize();
        p.setValues(source, photonDir, color);
        return p;        
    }
};

class DirectionalLight : public Light {
  public:
    Vector direction;
    Color color;
    
    DirectionalLight() {
        if (debug) cout << "Zero-parameter constructor of DirectionalLight is called." << endl;
        direction = Vector(); color = Color();
    }

    DirectionalLight(Vector d, Color c) {
        direction = d; color = c;
    }

    void generateLightRay(LocalGeo& local, Ray& lray, Color& lcolor) {
        lray.pos = local.pos;
        lray.dir = direction;
        lray.inv_dir = Vector(divide(1, lray.dir.x), divide(1, lray.dir.y), divide(1, lray.dir.z));
        lray.sign[0] = (lray.inv_dir.x < 0);
        lray.sign[1] = (lray.inv_dir.y < 0);
        lray.sign[2] = (lray.inv_dir.z < 0);
        lray.tMin = 0;
        lray.tMax = std::numeric_limits<float>::infinity();
        lcolor = color;
    }

    //type 1 indicates directional light
    int type() {
        return 1;
    }

    Point getSource() {
        cout << "DirectionalLight does not have a source." << endl;
        exit(1);
    }
    
    //no implementation for directional light implementation
    WPhoton emitPhoton() {
    }
};

class SquareLight : public Light {
  public:
    Point ul;
    Point ur;
    Point dl;
    Vector direction;
    Color color;
    
    SquareLight() {
        if (debug) cout << "Zero-parameter constructor of DirectionalLight is called." << endl;
        direction = Vector(); color = Color();
    }

    SquareLight(Point UL, Point UR, Point DL, Vector d, Color c) {
        ul = UL; ur = UR; dl = DL; direction = d; color = c;
    }

    void generateLightRay(LocalGeo& local, Ray& lray, Color& lcolor) {

        Point source = Point::addVector(ul, Vector::multiVS(Point::minusPoint(ur, ul), ((float)rand()/(RAND_MAX))));
        source = Point::addVector(source, Vector::multiVS(Point::minusPoint(dl, ul), ((float)rand()/(RAND_MAX))));

        Vector dir = Point::minusPoint(source, local.pos);
        lray.pos = local.pos;
        lray.dir = dir;
        lray.inv_dir = Vector(divide(1, lray.dir.x), divide(1, lray.dir.y), divide(1, lray.dir.z));
        lray.sign[0] = (lray.inv_dir.x < 0);
        lray.sign[1] = (lray.inv_dir.y < 0);
        lray.sign[2] = (lray.inv_dir.z < 0);
        lray.tMin = 0.00001;
        lray.tMax = 1;
        float i = -Vector::dotMul(Vector::normalizeV(direction), Vector::normalizeV(dir));
        if (i<=0){
            lcolor = Color();
        }else {
            lcolor = Color::multiplyS(color, i);
        }

    }

    //type 2 indicates square light
    int type() {
        return 2;
    }

    Point getSource() {
        Point source = Point::addVector(ul, Vector::multiVS(Point::minusPoint(ur, ul), ((float)rand()/(RAND_MAX))));
        source = Point::addVector(source, Vector::multiVS(Point::minusPoint(dl, ul), ((float)rand()/(RAND_MAX))));
        return source;
    }

    //use rejection sampling to emit photon
    WPhoton emitPhoton() {
        WPhoton p;
        Vector photonDir;
        Point photonPos = Point::addVector(ul, Vector::multiVS(Point::minusPoint(ur, ul), ((float)rand()/(RAND_MAX))));
        photonPos = Point::addVector(photonPos, Vector::multiVS(Point::minusPoint(dl, ul), ((float)rand()/(RAND_MAX))));
        do{
            photonDir.setValues(((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5, ((float)rand()/(RAND_MAX))-0.5);
            photonDir.normalize();
        } while(((float)rand()/(RAND_MAX)) >= Vector::dotMul(photonDir, direction));
        p.setValues(photonPos, photonDir, color);
        return p;        
    }
};
