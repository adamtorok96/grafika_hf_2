//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec3 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, vertexPosition.z, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

class vec3 {

public:
    float x, y, z;

    vec3() : x(0), y(0), z(0) {}

    vec3(float x, float y) {
        this->x = x;
        this->y = y;
        this->z = 0;
    }

    vec3(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    vec3(vec3 const & vec) {
        this->x = vec.x;
        this->y = vec.y;
        this->z = vec.z;
    }

    vec3 &operator=(vec3 const & vec) {
        this->x = vec.x;
        this->y = vec.y;
        this->z = vec.z;

        return *this;
    }

    const vec3 operator*(float n) const {
        return vec3(
                x * n,
                y * n,
                z * n
        );
    }

    vec3 operator/(float n) {
        if( n == 0 )
            throw std::exception();

        return vec3(
                x / n,
                y / n,
                z / n
        );
    }

    vec3 operator+(vec3 const & vec) const {
        return vec3(
                x + vec.x,
                y + vec.y,
                z + vec.z
        );
    }

    vec3 operator-(vec3 const & vec) {
        return vec3(
                x - vec.x,
                y - vec.y,
                z - vec.z
        );
    }

    vec3 operator*(vec3 const & vec) {
        return vec3(
                x * vec.x,
                y * vec.y,
                z * vec.z
        );
    }

    vec3 & operator+=(vec3 const & vec) {
        this->x += vec.x;
        this->y += vec.y;
        this->z += vec.z;

        return *this;
    }

    vec3 & operator*=(float n) {
        this->x *= n;
        this->y *= n;
        this->z *= n;

        return *this;
    }

    float dot(vec3 const & vec) {
        return x * vec.x + y * vec.y + z * vec.z;
    }

    float length() {
        return sqrtf(x * x + y * y + z * z);
    }

    vec3 & normalize() {
        float l = length();

        if( l != 0 ) {
            x /= l;
            y /= l;
            z /= l;
        }

        return *this;
    }

    vec3 cross(vec3 const & vec) {
        return vec3(
                y * vec.z - z * vec.y,
                z * vec.x - x * vec.z,
                x * vec.y - y * vec.x
        );
    }
};

// row-major matrix 4x4
struct mat4 {
    float m[4][4];

public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }

    mat4 operator*(const mat4& right) {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }

    static mat4 translate(vec3 const & pos) {
        return mat4(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                pos.x, pos.y, pos.z, 1
        );
    }

    static mat4 rotateX(float angle) {
        return mat4(
                1, 0, 0, 0,
                0, cosf(angle), -sinf(angle), 0,
                0, sinf(angle), cosf(angle), 0,
                0, 0, 0, 1
        );
    }

    static mat4 rotateY(float angle) {
        return mat4(
                cosf(angle), 0, -sinf(angle), 0, // - added to sin
                0, 1, 0, 0,
                sinf(angle), 0, cosf(angle), 0, // - removed from sin
                0, 0, 0, 1
        );
    }

    static mat4 rotateZ(float angle) {
        return mat4(
                cosf(angle), -sinf(angle), 0, 0,
                sinf(angle), cosf(angle), 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        );
    }

    operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
    float v[4];

    vec4(vec3 const & vec, float w = 1.0f) {
        v[0] = vec.x;
        v[1] = vec.y;
        v[2] = vec.z;
        v[3] = w;
    }

    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }

    vec4 operator*(const mat4& mat) {
        vec4 result;

        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;

            for (int i = 0; i < 4; i++)
                result.v[j] += v[i] * mat.m[i][j];
        }

        return result;
    }
};

// 2D camera
struct Camera {
    float wCx, wCy, wCz;	// center in world coordinates
    float wWx, wWy, wWz;	// width and height in world coordinates
public:
    Camera() : wCx(0), wCy(0), wCz(0), wWx(20), wWy(20), wWz(20) {}

    mat4 V() { // view matrix: translates the center to the origin
        return mat4(1,    0, 0, 0,
                    0,    1, 0, 0,
                    0,    0, 1, 0,
                    -wCx, -wCy, -wCz, 1);
    }

    mat4 P() { // projection matrix: scales it to be a square of edge length 2
        return mat4(2 / wWx,    0, 0, 0,
                    0,    2 / wWy, 0, 0,
                    0,        0, 2 / wWz, 0,
                    0,        0, 0, 1);
    }

    mat4 Vinv() { // inverse view matrix
        return mat4(1,     0, 0, 0,
                    0,     1, 0, 0,
                    0,     0, 1, 0,
                    wCx, wCy, wCz, 1);
    }

    mat4 Pinv() { // inverse projection matrix
        return mat4(wWx / 2, 0,    0, 0,
                    0, wWy / 2, 0, 0,
                    0,  0,    wWz / 2, 0,
                    0,  0,    0, 1);
    }
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
    unsigned int vao;	// vertex array object id
    float sx, sy;		// scaling
    float wTx, wTy;		// translation
public:
    Triangle() {
        Animate(0);
    }

    void Create() {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo[2];		// vertex buffer objects
        glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
        static float vertexCoords[] = { -8, -8, -6, 10, 8, -2 };	// vertex data on the CPU
        glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
                     sizeof(vertexCoords), // number of the vbo in bytes
                     vertexCoords,		   // address of the data array on the CPU
                     GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        // Data organization of Attribute Array 0
        glVertexAttribPointer(0,			// Attribute Array 0
                              2, GL_FLOAT,  // components/attribute, component type
                              GL_FALSE,		// not in fixed point format, do not normalized
                              0, NULL);     // stride and offset: it is tightly packed

        // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

        // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
        glEnableVertexAttribArray(1);  // Vertex position
        // Data organization of Attribute Array 1
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
    }

    void Animate(float t) {
        sx = 1; // sinf(t);
        sy = 1; // cosf(t);
        wTx = 0; // 4 * cosf(t / 2);
        wTy = 0; // 4 * sinf(t / 2);
    }

    void Draw() {
        mat4 Mscale(sx, 0, 0, 0,
                    0, sy, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 1); // model matrix

        mat4 Mtranslate(1,   0,  0, 0,
                        0,   1,  0, 0,
                        0,   0,  0, 0,
                        wTx, wTy,  0, 1); // model matrix

        mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

        // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
        int location = glGetUniformLocation(shaderProgram, "MVP");
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
        else printf("uniform MVP cannot be set\n");

        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
    }
};

class LineStrip {
    GLuint vao, vbo;        // vertex array object, vertex buffer object
    float  vertexData[120]; // interleaved data of coordinates and colors
    int    nVertices;       // number of vertices
public:
    LineStrip() {
        nVertices = 0;
    }
    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0
        glEnableVertexAttribArray(1);  // attribute array 1
        // Map attribute array 0 to the vertex data of the interleaved vbo
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
        // Map attribute array 1 to the color data of the interleaved vbo
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    }

    void AddPoint(float cX, float cY, float cZ) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        if (nVertices >= 20) return;

        vec4 wVertex = vec4(cX, cY, cZ, 1) * camera.Pinv() * camera.Vinv();
        // fill interleaved data
        vertexData[6 * nVertices]     = wVertex.v[0];
        vertexData[6 * nVertices + 1] = wVertex.v[1];
        vertexData[6 * nVertices + 2] = wVertex.v[2];
        vertexData[6 * nVertices + 3] = 1; // red
        vertexData[6 * nVertices + 4] = 1; // green
        vertexData[6 * nVertices + 5] = 0; // blue
        nVertices++;
        // copy data to the GPU
        glBufferData(GL_ARRAY_BUFFER, nVertices * 6 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
    }

    void Draw() {
        if (nVertices > 0) {
            mat4 VPTransform = camera.V() * camera.P();

            int location = glGetUniformLocation(shaderProgram, "MVP");
            if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
            else printf("uniform MVP cannot be set\n");

            glBindVertexArray(vao);
            glDrawArrays(GL_LINE_STRIP, 0, nVertices);
        }
    }
};

class Material {
public:
    vec3  F0;	// F0 spektrum
    float n;    // hullámhossz

    float ka; // ambient
    float kd; // diffuse
    float ks; // specular

    float shine;

    bool isRough, isReflective, isRefractive;

    vec3 color;

    Material & setF0(vec3 const & vec) {
        F0 = vec; return *this;
    }

    Material & setN(float n) {
        this->n = n; return *this;
    }

    vec3 reflect(vec3 inDir, vec3 normal) { // tükröző
        return inDir - normal * normal.dot(inDir) * 2.0f;
    }

    vec3 refract(vec3 inDir, vec3 normal) {  // törő
        float ior = n;

        float cosa = - normal.dot(inDir);

        if (cosa < 0) {
            cosa = -cosa;
            normal = normal * (-1);
            ior = 1 / n;

            float disc = 1 - (1 - cosa * cosa) / ior / ior;

            if (disc < 0)
                return reflect(inDir, normal); // total reflect

            return inDir / ior + normal * (cosa / ior - sqrt(disc));
        }
    }

    vec3 Fresnel(vec3 inDir, vec3 normal) { // közelítés
        float cosa = (float) fabs(normal.dot(inDir));

        return F0 + (vec3(1, 1, 1) - F0) * pow(1-cosa, 5);
    }

    vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad) {
        float cosTheta = normal.dot(lightDir);

        if(cosTheta < 0)
            return vec3(0, 0, 0);

        vec3 difRad = inRad * kd * cosTheta;
        vec3 halfway = (viewDir + lightDir).normalize();

        float cosDelta = normal.dot(halfway);

        if(cosDelta < 0)
            return difRad;

        return difRad + inRad * ks * pow(cosDelta, shine);
    }

    Material & setColor(vec3 const & vec) {
        color = vec; return *this;
    }

    void setReflective(bool i) {
        isReflective = i;
    }

    void setRefractive(bool i) {
        isRefractive = i;
    }
};

class Light {
public:
    vec3 pos, color;
};

class Ray {
public:
    vec3 pos, dir;

    Ray(vec3 const & position, vec3 const & direction) : pos(position), dir(direction) {}
};

struct Hit {
    float t;
    vec3 position;
    vec3 normal;
    Material * material;

    Hit() : t(-1), material(nullptr) {}
};

class Intersectable {
public:
    Material * material;

    virtual Hit intersect(const Ray & ray) = 0;
};

Hit firstIntersect(Intersectable ** objects, int n, Ray ray) {
    Hit bestHit;

    for(int i = 0; i < n; i++) {
        Hit hit = objects[i]->intersect(ray); //  hit.t < 0 if no intersection

        if(hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
            bestHit = hit;
    }

    return bestHit;
}

vec3 La;

float sign(float x) {
    if( x == 0.0f )
        return x;

    return x > 0.0f ? 1 : -1;
}

struct VertexData {
    vec3 position, normal;
    float u, v;
};

static float angleX = 15.0f, angleY = 5.0f, angleZ = 0.0f;
static float transX = 0.0f;

class ParamSurface : public Intersectable  {
    GLuint vao;
    GLuint vbo[2];

    unsigned int nVertices;

    vec3 color;

    VertexData * vtx;
    vec3 * vertexColors;

protected:
    virtual VertexData generateVertexData(float u, float v) = 0;

public:

    void setColor(vec3 const & color) {
        this->color = color;
    }

    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(2, &vbo[0]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        unsigned int N = 40;
        unsigned int M = N;
        nVertices = N * M * 6;

        vtx = new VertexData[nVertices];
        VertexData * pVtx = vtx;

        for (float i = 0.0f; i < N; i++) {
            for (float j = 0.0f; j < M; j++) {
                *pVtx++ = generateVertexData(i / N,        j / M);
                *pVtx++ = generateVertexData((i + 1) / N,  j / M);
                *pVtx++ = generateVertexData(i / N,        (j + 1) / M);
                *pVtx++ = generateVertexData((i + 1) / N,  j / M);
                *pVtx++ = generateVertexData((i + 1) / N,  (j + 1) / M);
                *pVtx++ = generateVertexData(i / N,        (j + 1) / M);
            }
        }

        glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), vtx, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);

        glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));

        glEnableVertexAttribArray(2);  // AttribArray 2 = UV
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, u));

        vertexColors = new vec3[nVertices];

        for(auto i = 0; i < nVertices; i++) {
            vertexColors[i] = color;
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * nVertices, vertexColors, GL_STATIC_DRAW);	// copy to the GPU

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        /*mat4 scale(
                1, 0, 0, 0,
                0, 0.5, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        );*/

        mat4 VPTransform =
                        mat4::translate(vec3(0, transX, 0)) *
                        mat4::rotateX(angleX) *
                        mat4::rotateY(angleY) *
                        mat4::rotateZ(angleZ) *
                        camera.V() *
                        camera.P();

       // angle += 0.06f;

        int location = glGetUniformLocation(shaderProgram, "MVP");

        if (location >= 0)
            glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
        else
            printf("uniform MVP cannot be set\n");

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, nVertices); // GL_LINE_STRIP GL_TRIANGLES
    }

    Hit intersect(Ray const & ray) override {
        Hit hit;

        for(int i = 0; i < nVertices; i += 3) {
            vec3 u = vtx[i + 1].position - vtx[i].position;
            vec3 v = vtx[i + 2].position - vtx[i].position;

            vec3 normal = u.cross(v);

            float D = normal.dot(vtx[i].position);

            float t = -(normal.dot(ray.pos) + D) / normal.dot(ray.dir);

            if( t < 0.0f )
                continue;

            // TODO: check .... is it in triangle?

            hit.t = t;
            hit.position = ray.pos + ray.dir * t;
            hit.material = material;
            hit.normal = normal;
        }

        return hit;
    }

    ~ParamSurface() {
        delete[] vtx;
        delete[] vertexColors;
    }
};

class Cylinder : public ParamSurface {
    float radius;

    VertexData generateVertexData(float u, float v) {
       /* vec3 normal = vec3(
                cosf(u * 2 * M_PI) * sinf(v *  M_PI),
                sinf(u * 2 * M_PI) * sinf(v *  M_PI),
                cosf(v * M_PI)
        );
   */
        vec3 position = vec3(
                radius * cosf(u * 2 * M_PI),
                radius * sinf(u * 2 * M_PI),
                v
        );

        vec4 wVertex = vec4(position, 1) * camera.Pinv() * camera.Vinv();

        return {
                vec3(wVertex.v[0], wVertex.v[1], wVertex.v[2]),
                vec3(),
                u, v
        };
    }

public:
    Cylinder(float rad = 0.0f) : radius(rad) {}

    void setRadius(float r) {
        radius = r;
    }

    float getRadius() { return radius; }
};

/*
 * bool rayTriangleIntersect(
    const Vec3f &orig, const Vec3f &dir,
    const Vec3f &v0, const Vec3f &v1, const Vec3f &v2,
    float &t)
{
    // compute plane's normal
    Vec3f v0v1 = v1 - v0;
    Vec3f v0v2 = v2 - v0;
    // no need to normalize
    Vec3f N = v0v1.crossProduct(v0v2); // N
    float area2 = N.length();

    // Step 1: finding P

    // check if ray and plane are parallel ?
    float NdotRayDirection = N.dotProduct(dir);
    if (fabs(NdotRayDirection) < kEpsilon) // almost 0
        return false; // they are parallel so they don't intersect !

    // compute d parameter using equation 2
    float d = N.dotProduct(v0);

    // compute t (equation 3)
    t = (N.dotProduct(orig) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    if (t < 0) return false; // the triangle is behind

    // compute the intersection point using equation 1
    Vec3f P = orig + t * dir;

    // Step 2: inside-outside test
    Vec3f C; // vector perpendicular to triangle's plane

    // edge 0
    Vec3f edge0 = v1 - v0;
    Vec3f vp0 = P - v0;
    C = edge0.crossProduct(vp0);
    if (N.dotProduct(C) < 0) return false; // P is on the right side

    // edge 1
    Vec3f edge1 = v2 - v1;
    Vec3f vp1 = P - v1;
    C = edge1.crossProduct(vp1);
    if (N.dotProduct(C) < 0)  return false; // P is on the right side

    // edge 2
    Vec3f edge2 = v0 - v2;
    Vec3f vp2 = P - v2;
    C = edge2.crossProduct(vp2);
    if (N.dotProduct(C) < 0) return false; // P is on the right side;

    return true; // this ray hits the triangle
} 
 */

vec3 trace(Intersectable ** objects, int n, Ray ray, unsigned int depth) {

    if (depth > 3)
        return La;

    Hit hit = firstIntersect(objects, n, ray);

    if(hit.t < 0)
        return La; // nothing

    //printf("HIT!\n");

    vec3 outRadiance = hit.material->color;

    /*
    if( hit.material->isRough ){
        outRadiance = La * hit.material->ka;

        for(each light source l){
            Ray shadowRay(r + N sign(N, V), Ll);
            Hit shadowHit = firstIntersect(objects, n, shadowRay);

            if(shadowHit.t < 0 || shadowHit.t > |r - yl| )
                outRadiance += hit.material->shade(hit.normal, hit.position, Ll, Lel);
        }
    }
    */


    if( hit.material->isReflective ){
        vec3 reflectionDir = hit.material->reflect(ray.dir, hit.normal);
        Ray reflectedRay(ray.dir + hit.normal * sign(hit.normal.dot(ray.dir)), reflectionDir); // r + N sign(NV)

        outRadiance += trace(objects, n, reflectedRay, depth + 1) * hit.material->Fresnel(hit.position, hit.normal);
    }

    if( hit.material->isRefractive ) {
        vec3 refractionDir = hit.material->refract(ray.dir, hit.normal);
        Ray refractedRay(ray.dir - hit.normal * sign(hit.normal.dot(ray.dir)), refractionDir); // r - N sign(NV)
        outRadiance += trace(objects, n, refractedRay, depth + 1) * (vec3(1,1,1) - hit.material->Fresnel(hit.position, hit.normal));
    }

    return outRadiance;
}

// The virtual world: collection of two objects
Cylinder **cyls; // = {0.5, 0.1};

Light light;

LineStrip lineStrip;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    cyls = new Cylinder*[2];

    cyls[0] = new Cylinder(0.5f);
    cyls[1] = new Cylinder(0.1f);

    cyls[0]->setRadius(0.5);
    cyls[1]->setRadius(0.1f);

    cyls[0]->setColor(vec3(218.0f / 255, 165.0f / 255, 32.0f / 255));
    cyls[1]->setColor(vec3(200.0f / 255, 100.0f / 255, 32.0f / 255));

    cyls[0]->Create();
    cyls[1]->Create();

    cyls[0]->material = new Material();
    cyls[0]->material->setColor(vec3(218.0f / 255, 165.0f / 255, 32.0f / 255));
    cyls[0]->material->setReflective(false);
    cyls[0]->material->setRefractive(false);
    cyls[0]->material->setF0(vec3(0.17f/3.1f, 0.35f/2.7f, 1.5f/1.9f));

    cyls[1]->material = new Material();
    cyls[1]->material->setColor(vec3(218.0f / 255, 165.0f / 255, 32.0f / 255));
    cyls[1]->material->setReflective(false);
    cyls[1]->material->setRefractive(false);
    cyls[1]->material->setF0(vec3(0.17f/3.1f, 0.35f/2.7f, 1.5f/1.9f));

    //Ray ray(vec3(1.0, 0.0, 0.0), vec3(-1.0f, 0.0, 0.0));

    //vec3 color = trace(&cyl, 1, ray, 0);

    //printf("%f %f %f\n", color.x, color.y, color.z);

#if true
    FILE * f = fopen("img.ppm", "w+");
    fprintf(f, "P3\n%d %d\n255\n", windowWidth, windowHeight);

    for(int h = 0; h < windowHeight; h++) {
        printf("h: %d / %d\n", h, windowHeight);
        for(int w = 0; w < windowWidth; w++) {
            Ray ray(
                    vec3(
                            (w / windowWidth) * 2.0f - 1.0f,
                            (h / windowHeight) * 2.0f - 1.0f,
                            0.0f
                    ),
                    vec3(1, 0, 0)
            );

            vec3 color = trace(reinterpret_cast<Intersectable**>(cyls), 2, ray, 0);

            fprintf(f, "%d %d %d ", (int)(color.x * 255), (int)(color.y * 255), (int)(color.z * 255));
        }
    }

    fclose(f);
#endif
    lineStrip.Create();
    lineStrip.AddPoint(0.1f, -0.5f, -0.5f);
    lineStrip.AddPoint(-0.8f, 0.5f, 0.7f);

    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");

    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");

    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

    // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);

    glEnable(GL_DEPTH);
    glEnable(GL_DEPTH_TEST);
}

void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

    cyls[1]->Draw();
    cyls[0]->Draw();

    lineStrip.Draw();

    glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'r')
        glutPostRedisplay();         // if d, invalidate display, i.e. redraw
    else if( key == 'd' )
        angleX += 0.01;
    else if( key == 'a' )
        angleX -= 0.01;
    else if( key == 'w' )
        angleY += 0.01;
    else if( key == 's' )
        angleY -= 0.01;
    else if( key == 'm' )
        angleZ += 0.01;
    else if( key == 'n' )
        angleZ -= 0.01;
    else {
        switch(key) {
            case 'l': transX += 0.1f; break;
            case 'k': transX -= 0.1f; break;
        }
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
        float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;

        glutPostRedisplay();     // redraw
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    float sec = time / 1000.0f;				// convert msec to sec

    glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
    glewExperimental = true;	// magic
    glewInit();
#endif

    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    onInitialization();

    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    onExit();
    return 1;
}
