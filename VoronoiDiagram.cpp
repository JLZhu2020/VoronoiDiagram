
#if 1

#define GLEW_STATIC
#define STB_IMAGE_IMPLEMENTATION

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <map>
#include <unordered_map>
#include <random>
#include <iostream>
//#include <windows.h>

using namespace std;

void generatePointSet(double** data_vertices, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disx(-1, 1);
    for (int i = 0; i < size; i++) {
        data_vertices[i] = new double[2];
        data_vertices[i][0] = disx(gen);
        data_vertices[i][1] = disx(gen);
    }
    double x_min = 1;
    double x_max = -1;
    double y_min = 1;
    double y_max = -1;
    for (int i = 0; i < size; i++) {
        x_min = min(x_min, data_vertices[i][0]);
        x_max = max(x_max, data_vertices[i][0]);
        y_min = min(y_min, data_vertices[i][1]);
        y_max = max(y_max, data_vertices[i][1]);
    }
    for (int i = 0; i < size; i++) {
        data_vertices[i][0] = 1.8 * (data_vertices[i][0] - x_min) / (x_max - x_min) - 0.9;
        data_vertices[i][1] = 1.8 * (data_vertices[i][1] - y_min) / (y_max - y_min) - 0.9;
    }
}

void midPrepend(double* point1, double* point2, double* mid, double* another) {
    mid[0] = (point1[0] + point2[0]) / 2;
    mid[1] = (point1[1] + point2[1]) / 2;
    if (point1[0] < point2[0]) {
        another[0] = mid[0] + (point2[1] - point1[1]) / 2;
        another[1] = mid[1] - (point2[0] - point1[0]) / 2;
    }
    else {
        another[0] = mid[0] - (point2[1] - point1[1]) / 2;
        another[1] = mid[1] + (point2[0] - point1[0]) / 2;
    }
}

class Edge {
public:
    int id;
    static int count;
    double* p1;
    double* p2;
    int* known;
    int state;  //0: 2 points unknown, 1: 1 point unknown, 2: two points known
    int next;
    int pre;
    Edge() {
        this->id = count;
        count++;
        this->p1 = new double[2];
        this->p2 = new double[2];
        this->known = new int[2];
        this->known[0] = 0;
        this->known[1] = 0;
        this->state = 0;
        this->pre = -1;
        this->next = -1;
    }
    Edge(double* p) :Edge() {
        this->p1[0] = p[0];
        this->p1[1] = p[1];
        this->p2[0] = p[0];
        this->p2[1] = p[1];
    }
    Edge(double* p1, double* p2) :Edge() {
        if (p1[0] < p2[0]) {
            this->p1[0] = p1[0];
            this->p1[1] = p1[1];
            this->p2[0] = p2[0];
            this->p2[1] = p2[1];
        }
        else {
            this->p1[0] = p2[0];
            this->p1[1] = p2[1];
            this->p2[0] = p1[0];
            this->p2[1] = p1[1];
        }

    }
    void updatePoint(double* p1, double* p2) {
        if (p2[0] < p1[0]) {
            double* t = new double[2];
            t[0] = p1[0];
            t[1] = p1[1];
            p1[0] = p2[0];
            p1[1] = p2[1];
            p2[0] = t[0];
            p2[1] = t[1];
        }
        if (this->known[0] == 0) {
            this->p1[0] = p1[0];
            this->p1[1] = p1[1];
        }
        if (this->known[1] == 0) {
            this->p2[0] = p2[0];
            this->p2[1] = p2[1];
        }
    }
    void setPoint(double* p) {
        if (p[0] < p1[0]) {
            p1[0] = p[0];
            p1[1] = p[1];
            known[0] = 1;
        }
        else {
            p2[0] = p[0];
            p2[1] = p[1];
            known[1] = 1;
        }
        this->state++;
    }
    void setPre(int pre) {
        this->pre = pre;
    }
    void setNext(int next) {
        this->next = next;
    }
    bool operator==(const Edge& another)const {
        return another.p1[0] == this->p1[0] &&
            another.p1[1] == this->p1[1] &&
            another.p2[0] == this->p2[0] &&
            another.p2[1] == this->p2[1];
    }
    void extendToBoundary() {
        double trend = (p1[1] - p2[1]) / (p1[0] - p2[0]);
        if (known[0] == 1) {
            p2[0] = 1;
            p2[1] = p1[1] + (1 - p1[0]) * trend;
        }
        else {
            p1[0] = -1;
            p1[1] = p2[1] - (p2[0] + 1) * trend;
        }
    }
    void setKnown(int who) {
        this->known[who] = 1;
    }
    void print() {
        cout << " edge id : " << this->id << " state " << this->state << " known " << this->known[0] << " " << this->known[1] << endl;
        cout << this->p1[0] << " " << this->p1[1] << endl;
        cout << this->p2[0] << " " << this->p2[1] << endl;
        cout << endl;
    }
};

int Edge::count = 0;

class Face {
public:
    int id;
    static int count;
    double* coordinate;
    int edge;
    Face() {
        this->id = count;
        count++;
        this->coordinate = new double[2];
        this->edge = -1;
    }
    Face(double* coordinate) :Face() {
        this->coordinate[0] = coordinate[0];
        this->coordinate[1] = coordinate[1];
    }
    void setEdge(int edge) {
        this->edge = edge;
    }
    void print() {
        cout << this->coordinate[0] << " " << this->coordinate[1] << endl;
    }
};

int Face::count = 0;

class Event {
public:
    int id;
    static int count;
    double y;
    double* circleCenter;
    int face;
    int edge1;
    int edge2;
    int type;  //type 1 is point event, type 0 is circle event
    int arc;
    Event() {
        this->id = count;
        count++;
        this->circleCenter = new double[2];
        this->face = -1;
        this->y = -1;
        this->type = -1;
        this->arc = -1;
        this->edge1 = -1;
        this->edge2 = -1;
    };
    Event(int face, double y) : Event() {
        this->y = y;
        this->face = face;
        this->type = 1;
    }
    Event(double* circleCenter, double y, int edge1, int edge2, int arc) : Event() {
        this->circleCenter[0] = circleCenter[0];
        this->circleCenter[1] = circleCenter[1];
        this->y = y;
        this->type = 0;
        this->edge1 = edge1;
        this->edge2 = edge2;
        this->arc = arc;
    }
    bool operator==(const Event& another)const {
        return another.y == this->y && another.id == this->id;
    }
};

int Event::count = 0;

class Arc {
public:
    int id;
    static int count;
    int face;
    int evId;
    double intersect;
    int pre;
    int next;
    unordered_map<int, int >edgeMap;
    Arc() {
        this->id = count;
        count++;
        this->evId = -1;
        this->intersect = 1.0;
        this->pre = -1;
        this->next = -1;
        this->edgeMap = unordered_map<int, int >();
    }
    Arc(int face, double intersect) :Arc() {
        this->face = face;
        this->intersect = intersect;
    }
    Arc(int face, int evId, double intersect) :Arc() {
        this->face = face;
        this->evId = evId;
        this->intersect = intersect;
    }
    Arc(int face, int evId, double intersect, int pre, int next) :Arc() {
        this->face = face;
        this->evId = evId;
        this->intersect = intersect;
        this->pre = pre;
        this->next = next;
    }
    void setEvent(int evId) {
        this->evId = evId;
    }
    void setPre(int pre) {
        this->pre = pre;
    }
    void setNext(int next) {
        this->next = next;
    }
    void setPN(int pre, int next) {
        this->pre = pre;
        this->next = next;
    }
    bool operator==(const Arc& another)const {
        return another.intersect == this->intersect;
    }
};

int Arc::count = 0;

int comperator(Event a, Event b) {
    double dif = a.y - b.y;
    if (dif > 0)return -1;
    return 1;
}

namespace std {
    template <>
    struct hash<Event>
    {
        std::size_t operator()(const Event& k) const
        {
            using std::size_t;
            using std::hash;
            return hash<double>()(k.y);
        }
    };
    template <>
    struct hash<Arc>
    {
        std::size_t operator()(const Arc& k) const
        {
            using std::size_t;
            using std::hash;
            return hash<double>()(k.intersect);
        }
    };
    template <>
    struct hash<Edge>
    {
        std::size_t operator()(const Edge& k) const
        {
            using std::size_t;
            using std::hash;
            return hash<double>()(k.p1[0]);
        }
    };
}

template<typename T> class Heap {
public:
    vector<T>heap;
    unordered_map<T, int>position;
    int size;
    int (*comperator)(T, T);
    Heap(int (*comperator)(T, T)) {
        this->size = 0;
        this->comperator = comperator;
    }
    void swap(T key1, T key2) {
        int temp = position[key1];
        this->position[key1] = this->position[key2];
        position[key2] = temp;
    }
    void add(T val) {
        this->heap.push_back(val);
        int pin = this->size;
        position[val] = pin;
        while (pin > 0 && comperator(heap[(pin - 1) / 2], val) > 0) {
            swap(heap[pin], heap[(pin - 1) / 2]);
            heap[pin] = heap[(pin - 1) / 2];
            pin = (pin - 1) / 2;
            heap[pin] = val;
        }
        this->size++;
    }
    T pop() {
        T res = heap[0];
        pop(res);
        return res;
    }
    bool inHeap(T val) {
        auto it = position.find(val);
        return it != position.end();
    }
    void pop(T val) {
        int pin = position[val];
        position.erase(val);
        this->size--;
        this->heap[pin] = this->heap[this->size];
        position[heap[pin]] = pin;
        this->heap.pop_back();
        if (pin == this->size)return;
        
        while (pin > 0 && comperator(heap[(pin - 1) / 2], heap[pin]) > 0) {
            swap(heap[pin], heap[(pin - 1) / 2]);
            T temp = this->heap[pin];
            heap[pin] = heap[(pin - 1) / 2];
            pin = (pin - 1) / 2;
            heap[pin] = temp;
        }
        while (true) {
            if (pin * 2 + 2 < this->size &&
                (this->comperator(this->heap[pin], this->heap[pin * 2 + 1]) > 0 || this->comperator(this->heap[pin], this->heap[pin * 2 + 2]) > 0)) {
                if (comperator(this->heap[pin * 2 + 1], this->heap[pin * 2 + 2]) < 0) {
                    T temp = this->heap[pin];
                    this->heap[pin] = this->heap[pin * 2 + 1];
                    this->heap[pin * 2 + 1] = temp;
                    swap(heap[pin], heap[pin * 2 + 1]);
                    pin = pin * 2 + 1;
                }
                else {
                    T temp = this->heap[pin];
                    this->heap[pin] = this->heap[pin * 2 + 2];
                    this->heap[pin * 2 + 2] = temp;
                    swap(heap[pin], heap[pin * 2 + 2]);
                    pin = pin * 2 + 2;
                }
            }
            else if (pin * 2 + 2 == this->size && this->comperator(this->heap[pin], this->heap[pin * 2 + 1]) > 0) {
                T temp = this->heap[pin];
                this->heap[pin] = this->heap[pin * 2 + 1];
                this->heap[pin * 2 + 1] = temp;
                swap(heap[pin], heap[pin * 2 + 1]);
                pin = pin * 2 + 1;
            }
            else break;
        }
    }
    void prin() {
        for (int i = 0; i < this->size; i++) {
            int k = position[heap[i]];
            cout << "" << k << " ";
        }
        cout << endl;
    }
};

double* twoLineIntersect(double* point1, double* point2, double* point3, double* point4) {
    double* res = (double*)malloc(2 * sizeof(double));
    if (point1[0] == point2[0])point1[0] += 0.0000001;
    if (point3[0] == point4[0])point3[0] += 0.0000001;
    double a1 = (point1[1] - point2[1]) / (point1[0] - point2[0]);
    double b1 = (point2[1] * point1[0] - point1[1] * point2[0]) / (point1[0] - point2[0]);
    double a2 = (point3[1] - point4[1]) / (point3[0] - point4[0]);
    double b2 = (point4[1] * point3[0] - point3[1] * point4[0]) / (point3[0] - point4[0]);
    res[0] = (b2 - b1) / (a1 - a2);
    res[1] = (a1 * b2 - a2 * b1) / (a1 - a2);
    return res;
}

double* getCircleCenter(double* point1, double* point2, double* point3) {
    double* mid1 = new double[2];
    double* another1 = new double[2];
    double* mid2 = new double[2];
    double* another2 = new double[2];
    midPrepend(point1, point2, mid1, another1);
    midPrepend(point1, point3, mid2, another2);
    return twoLineIntersect(mid1, another1, mid2, another2);
}

void parabolaIntersect(double* point1, double* point2, double directrix, double& intersect1, double& intersect2) {
    double a = (point1[1] - point2[1]);
    double b = 2 * (point2[0] * (directrix - point1[1]) - point1[0] * (directrix - point2[1]));
    double c = (point1[1] - directrix) * point2[0] * point2[0] - (point2[1] - directrix) * point1[0] * point1[0] -
        (point1[1] - point2[1]) * (point2[1] - directrix) * (point1[1] - directrix);
    intersect1 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
    intersect2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
}

void updateBeachLine(map<double, int>& beachLine, map<double, int>& newBeachLine,
    vector<Arc>& arcs, vector<Face>faces, double y) {

    for (auto it : beachLine) {
        Arc& arc = arcs[it.second];
        if (arc.next >= 0) {
            Arc next = arcs[arc.next];
            double inter1, inter2;
            double* point1 = faces[arc.face].coordinate;
            double* point2 = faces[next.face].coordinate;
            parabolaIntersect(point1, point2, y, inter1, inter2);
            double sep = (point1[1] < point2[1] ? point1[0] : point2[0]);
            if (inter1 > inter2) {
                double temp = inter1;
                inter1 = inter2;
                inter2 = temp;
            }
            double inter = (arc.intersect < sep ? inter1 : inter2);
            newBeachLine[inter] = arc.id;
            arc.intersect = inter;
        }
        else newBeachLine[1e10] = arc.id;

    }
}

void edgeInit(double* point1, double* point2, double directrix, double* intersect1, double* intersect2) {
    double inter1, inter2;
    parabolaIntersect(point1, point2, directrix, inter1, inter2);
    intersect1[0] = inter1;
    intersect1[1] = (inter1 - point1[0]) * (inter1 - point1[0]) / (2 * (point1[1] - directrix)) + (point1[1] + directrix) / 2;
    intersect2[0] = inter2;
    intersect2[1] = (inter2 - point2[0]) * (inter2 - point2[0]) / (2 * (point2[1] - directrix)) + (point2[1] + directrix) / 2;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main() {
    int size = 100;
    double** data = new double* [size];
    generatePointSet(data, size);

    for (int i = 0; i < size; i++)cout << "data[" << i << "]=new double[]{" << data[i][0] << "," << data[i][1] << "};" << endl;


    map<double, int>beachline;
    vector<Face> faces;
    vector<Arc> arcs;
    vector<Edge >edges;
    vector<Event >events;
    Heap<Event>eventQueue(comperator);
    for (int i = 0; i < size; i++) {
        Face face(data[i]);
        faces.push_back(face);
        Event ev(i, data[i][1]);
        events.push_back(ev);
        eventQueue.add(ev);
    }
    double deta = 0.0000000001;
    double left = -1;
    double right = 1e10;
    Event ev = eventQueue.pop();
    Arc arc(ev.face, right);
    arcs.push_back(arc);
    beachline[right] = arc.id;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(600, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glViewport(0, 0, 600, 600);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    const char* vertexShaderSource = "#version 330 core\n"
        "layout (location = 0) in vec2 aPos;\n"
        "void main()\n"
        "{\n"
        "   gl_Position = vec4(aPos, 0.0, 1.0);\n"
        "}\0";

    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    const char* fragmentShaderSource = "#version 330 core\n"
        //  "uniform vec4 ourColor;\n"
        "uniform vec3 color;\n"
        "out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
        "   FragColor = vec4(color,1.0);\n"
        "}\n\0";

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    unsigned int VBO;
    glGenBuffers(1, &VBO);

    unsigned int EBO;
    glGenBuffers(1, &EBO);

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, 2 * sizeof(double), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glPointSize(6);

    double* pointData = new double[2 * size];
    for (int i = 0; i < size; i++) {
        pointData[2 * i] = data[i][0];
        pointData[2 * i + 1] = data[i][1];
    }

    for (double scanLine = 1; scanLine > -1.05 || eventQueue.size > 0|| !glfwWindowShouldClose(window); scanLine -= 0.001/*(scanLine>-1.05?0.01:10)*/ ) {
        processInput(window);
        if (scanLine > -2 || eventQueue.size > 0) {
            while (eventQueue.size > 0 && scanLine < eventQueue.heap[0].y) {
                Event ev = eventQueue.pop();
                double y = ev.y;
                if (ev.type == 1) {
                    map<double, int>newBeachLine;
                    updateBeachLine(beachline, newBeachLine, arcs, faces, y);
                    beachline.clear();
                    beachline = newBeachLine;
                    Face f = faces[ev.face];
                    Arc& rarc = arcs[beachline.upper_bound(f.coordinate[0] + 2 * deta)->second];
                    Face oldf = faces[rarc.face];
                    double inter1 = f.coordinate[0] - deta;
                    if (rarc.pre >= 0 && arcs[rarc.pre].intersect >= inter1)inter1 = (f.coordinate[0] + arcs[rarc.pre].intersect) / 2;
                    double inter2 = f.coordinate[0] + deta;
                    if (inter2 >= rarc.intersect)inter2 = (f.coordinate[0] + rarc.intersect) / 2;
                    Arc larc(oldf.id, inter1);
                    Arc marc(f.id, inter2);
                    if (rarc.pre >= 0) {
                        Arc& parc = arcs[rarc.pre];
                        parc.setNext(larc.id);
                        parc.edgeMap[larc.id] = parc.edgeMap[rarc.id];
                        larc.edgeMap[parc.id] = rarc.edgeMap[parc.id];
                    }
                    larc.setPN(rarc.pre, marc.id);
                    marc.setPN(larc.id, rarc.id);
                    rarc.setPre(marc.id);
                    double* endp1 = new double[2];
                    double* endp2 = new double[2];
                    edgeInit(f.coordinate, faces[rarc.face].coordinate, y - deta, endp1, endp2);
                    Edge edge(endp1, endp2);
                    edges.push_back(edge);
                    larc.edgeMap[marc.id] = edge.id;
                    marc.edgeMap[larc.id] = edge.id;
                    marc.edgeMap[rarc.id] = edge.id;
                    rarc.edgeMap[marc.id] = edge.id;
                    if (rarc.evId >= 0) {
                        Event evv = events[rarc.evId];
                        if (eventQueue.inHeap(evv))eventQueue.pop(evv);
                        rarc.evId = -1;
                    }

                    if (larc.pre >= 0) {
                        Arc& lpre = arcs[larc.pre];
                        double* circleCenter = getCircleCenter(
                            faces[lpre.face].coordinate, faces[larc.face].coordinate, f.coordinate);
                        if (circleCenter[0] < f.coordinate[0]) {
                            double r = sqrt((circleCenter[0] - f.coordinate[0]) * (circleCenter[0] - f.coordinate[0]) +
                                (circleCenter[1] - f.coordinate[1]) * (circleCenter[1] - f.coordinate[1]));
                            Event eve(circleCenter, circleCenter[1] - r, larc.edgeMap[larc.pre], larc.edgeMap[larc.next], larc.id);
                            events.push_back(eve);
                            eventQueue.add(eve);
                            larc.evId = eve.id;
                        }
                    }
                    if (rarc.next >= 0) {
                        Arc& rnext = arcs[rarc.next];
                        double* circleCenter = getCircleCenter(
                            faces[rnext.face].coordinate, faces[rarc.face].coordinate, f.coordinate);
                        if (circleCenter[0] > f.coordinate[0]) {
                            double r = sqrt((circleCenter[0] - f.coordinate[0]) * (circleCenter[0] - f.coordinate[0]) +
                                (circleCenter[1] - f.coordinate[1]) * (circleCenter[1] - f.coordinate[1]));
                            Event eve(circleCenter, circleCenter[1] - r, rarc.edgeMap[rarc.pre], rarc.edgeMap[rarc.next], rarc.id);
                            events.push_back(eve);
                            eventQueue.add(eve);
                            rarc.evId = eve.id;
                        }
                    }
                    arcs.push_back(larc);
                    arcs.push_back(marc);
                    beachline[larc.intersect] = larc.id;
                    beachline[marc.intersect] = marc.id;
                }
                else {
                    Arc& darc = arcs[ev.arc];
                    Arc& pre = arcs[darc.pre];
                    Arc& next = arcs[darc.next];
                    Edge& e1 = edges[ev.edge1];
                    Edge& e2 = edges[ev.edge2];
                    e1.setPoint(ev.circleCenter);
                    e2.setPoint(ev.circleCenter);
                    Edge edge(ev.circleCenter);
                    edge.state = 1;
                    if (ev.circleCenter[0] < -1 || ev.circleCenter[0]>1)edge.state = 2;
                    double* inter1 = new double[2];
                    double* inter2 = new double[2];
                    double* inter3 = new double[2];
                    double* inter4 = new double[2];
                    edgeInit(faces[pre.face].coordinate, faces[next.face].coordinate, y, inter1, inter2);
                    edgeInit(faces[pre.face].coordinate, faces[next.face].coordinate, y - deta, inter3, inter4);
                    if (abs(ev.circleCenter[0] - inter1[0]) < 0.000001) {
                        if (inter3[0] < inter1[0]) {
                            edge.setKnown(1);
                        }
                        else edge.setKnown(0);
                        edge.updatePoint(inter3, inter4);
                    }
                    else {
                        if (inter4[0] < inter2[0]) {
                            edge.setKnown(1);
                        }
                        else edge.setKnown(0);
                        edge.updatePoint(inter3, inter4);
                    }
                    edges.push_back(edge);
                    pre.edgeMap[next.id] = edge.id;
                    next.edgeMap[pre.id] = edge.id;
                    pre.setNext(next.id);
                    next.setPre(pre.id);
                    pre.next = next.id;
                    next.pre = pre.id;
                    beachline.erase(darc.intersect);
                    if (pre.evId >= 0) {
                        Event& eev = events[pre.evId];
                        if (eventQueue.inHeap(eev))eventQueue.pop(eev);
                        pre.evId = -1;
                    }
                    if (next.evId >= 0) {
                        Event& eev = events[next.evId];
                        if (eventQueue.inHeap(eev))eventQueue.pop(eev);
                        next.evId = -1;
                    }
                    if (pre.pre >= 0) {
                        Arc& ppre = arcs[pre.pre];
                        double* circleCenter = getCircleCenter(
                            faces[pre.face].coordinate, faces[ppre.face].coordinate, faces[next.face].coordinate);
                        double r = sqrt((circleCenter[0] - faces[next.face].coordinate[0]) * (circleCenter[0] - faces[next.face].coordinate[0]) +
                            (circleCenter[1] - faces[next.face].coordinate[1]) * (circleCenter[1] - faces[next.face].coordinate[1]));
                        Face& fpp = faces[ppre.face];
                        Face& fp = faces[pre.face];
                        Face& fn = faces[next.face];
                        bool cond1 = false, cond2 = false;
                        double lowx1 = (fpp.coordinate[1] < fp.coordinate[1] ? fpp.coordinate[0] : fp.coordinate[0]);
                        double lowx2 = (fp.coordinate[1] < fn.coordinate[1] ? fp.coordinate[0] : fn.coordinate[0]);
                        cond1 = (circleCenter[0] < lowx1 && ppre.intersect < lowx1) || (circleCenter[0] > lowx1 && ppre.intersect > lowx1);
                        cond2 = (circleCenter[0] < lowx2 && pre.intersect < lowx2) || (circleCenter[0] > lowx2 && pre.intersect > lowx2);

                        if ((circleCenter[1] - r < y) && cond1 && cond2) {
                            Event eve(circleCenter, circleCenter[1] - r, pre.edgeMap[pre.pre], pre.edgeMap[pre.next], pre.id);
                            events.push_back(eve);
                            eventQueue.add(eve);
                            pre.evId = eve.id;
                        }
                    }
                    if (next.next >= 0) {
                        Arc& nnext = arcs[next.next];
                        double* circleCenter = getCircleCenter(
                            faces[pre.face].coordinate, faces[next.face].coordinate, faces[nnext.face].coordinate);
                        double r = sqrt((circleCenter[0] - faces[next.face].coordinate[0]) * (circleCenter[0] - faces[next.face].coordinate[0]) +
                            (circleCenter[1] - faces[next.face].coordinate[1]) * (circleCenter[1] - faces[next.face].coordinate[1]));
                        Face& fp = faces[pre.face];
                        Face& fn = faces[next.face];
                        Face& fnn = faces[nnext.face];
                        bool cond1 = false, cond2 = false;
                        double lowx1 = (fp.coordinate[1] < fn.coordinate[1] ? fp.coordinate[0] : fn.coordinate[0]);
                        double lowx2 = (fn.coordinate[1] < fnn.coordinate[1] ? fn.coordinate[0] : fnn.coordinate[0]);
                        cond1 = (circleCenter[0] < lowx1 && pre.intersect < lowx1) || (circleCenter[0] > lowx1 && pre.intersect > lowx1);
                        cond2 = (circleCenter[0] < lowx2 && next.intersect < lowx2) || (circleCenter[0] > lowx2 && next.intersect > lowx2);

                        if ((circleCenter[1] - r < y) && cond1 && cond2) {
                            Event eve(circleCenter, circleCenter[1] - r, next.edgeMap[next.pre], next.edgeMap[next.next], next.id);
                            events.push_back(eve);
                            eventQueue.add(eve);
                            next.evId = eve.id;
                        }
                    }

                }
            }

            for (auto it : beachline) {
                Arc& arcc = arcs[it.second];
                if (arcc.next >= 0) {
                    Edge& edge = edges[arcc.edgeMap[arcc.next]];
                    if (edge.state < 2) {
                        double* inter1 = new double[2];
                        double* inter2 = new double[2];
                        edgeInit(faces[arcc.face].coordinate, faces[arcs[arcc.next].face].coordinate, scanLine - deta, inter1, inter2);
                        // cout << scanLine<<" " << inter1[0] << " " << inter1[1] << " " << inter2[0] << " " << inter2[1] << endl;
                        // cout << endl;
                        edge.updatePoint(inter1, inter2);
                        //edge.print();

                    }
                }
            }

            int edge_num = edges.size();

            double* lineData = new double[edge_num * 4];
            for (int i = 0; i < edge_num; i++) {
                Edge& edge = edges[i];
                /*if (edge.state == 0) {
                    for (int j = 0; j < 4; j++)lineData[4 * i + j] = 0;
                    continue;
                }
                if (edge.state == 1) {
                    edge.extendToBoundary();
                }*/
                lineData[4 * i] = edge.p1[0];
                lineData[4 * i + 1] = edge.p1[1];
                lineData[4 * i + 2] = edge.p2[0];
                lineData[4 * i + 3] = edge.p2[1];
            }

            int* indices = new int[2 * edge_num];
            for (int i = 0; i < 2 * edge_num; i++)indices[i] = i;

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(shaderProgram);
            glBindVertexArray(VAO);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * edge_num * sizeof(int), indices, GL_STATIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 0.0f, 0.0f);
            glBufferData(GL_ARRAY_BUFFER, 2 * size * sizeof(double), pointData, GL_STATIC_DRAW);
            glDrawArrays(GL_POINTS, 0, size);

            glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 1.0f, 1.0f);
            glBufferData(GL_ARRAY_BUFFER, 4 * edge_num * sizeof(double), lineData, GL_STATIC_DRAW);
            glDrawElements(GL_LINES, 2 * edge_num, GL_UNSIGNED_INT, 0);
            //Sleep(50);
        }      
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    //while (!glfwWindowShouldClose(window))
    //{
    //    processInput(window);
    //    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    //    glClear(GL_COLOR_BUFFER_BIT);

    //    glUseProgram(shaderProgram);
    //    glBindVertexArray(VAO);

    //    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    //    glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 0.0f, 0.0f);
    //    glBufferData(GL_ARRAY_BUFFER, 2 * size * sizeof(double), pointData, GL_STATIC_DRAW);
    //    glDrawArrays(GL_POINTS, 0, size);

    //    glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 1.0f, 1.0f);
    //    glBufferData(GL_ARRAY_BUFFER, 4 * edge_num * sizeof(double), lineData, GL_STATIC_DRAW);
    //    glDrawElements(GL_LINES, 2 * edge_num, GL_UNSIGNED_INT, 0);
    //    

    //    glfwSwapBuffers(window);
    //    glfwPollEvents();
    //}
    glfwTerminate();

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    return 0;
}

#endif