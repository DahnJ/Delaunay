
#include <iostream>
#include <cmath>
#include <algorithm>  // for min and max
#include <vector>
#include <numeric>    // For sum of vector
#include <functional> //
#include <iterator>
#include <random>
#include <atomic>
#include <set>
#include <unordered_map>
#include <queue>
#include <ctime>



bool VERBOSE;
bool VERBOSE2;
bool VERBOSE3;
bool VERBOSE4;
bool VERBOSE5;
bool POINT_INDEX_ONLY;

// NOTES
// TODO Classes in separate files
// TODO Extend the code for power diagrams


// TODO http://openframeworks.cc/documentation/math/ofVec3f/


// TODOS

// Circumsphere size for tetrahedron
// find neighbors of a point
// sew in method
// actual incremental

// TODO: add consts where they belong (e.g. functions)
// TODO: Multiple distances with polymorphism?

//// Abstract class shape that will contain all possible facets
//// Useful for e.g. list of a certain k-facet?
//class Shape {
//public:
//	virtual ~Shape() {};
//};



// Learn about: templates, iterators, copy
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
	if (!v.empty()) {
		out << '[';
		std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
		out << "\b\b]";
	}
	return out;
}


template <typename T>
bool isIn(const std::vector<T>& v, const T& e) {
	return(std::find(v.begin(), v.end(), e) != v.end());
}



template <typename T>
std::ostream& operator<< (std::ostream& out, const std::set<T>& s) {
	if (!s.empty()) {
		out << '[';
		for (T t : s) {
			out << t;
		};
		out << "]";
	}
	return out;
}




class Point {
private:
	std::vector<double> coords;
	int id;

	std::vector<Point> neighbors;

	static std::random_device seed_generator;
	static unsigned seed;
	static std::mt19937 mersenne_generator;
	static std::uniform_real_distribution<double> distribution;

public:
	// static std::atomic<int> s_id;

	Point(double _x = 0.0, double _y = 0.0, double _z = 0.0, int _id = -1) : id(_id) {
		coords.push_back(_x);
		coords.push_back(_y);
		coords.push_back(_z);
		// std::cout << "Creating a point" << _x << _y << _z <<_id << std::endl;
	};
	Point(std::vector<double> _coords, int _id = -1) : coords(_coords), id(_id) {
	};
	// Point(const Point& other) : x(other.getX()), y(other.getY()), z(other.getZ) {};   No need for a copy constructor - it is constructed automatically

	static Point randomizePoint(int _id = -1) {
		std::cout << seed << std::endl;
		return Point(distribution(mersenne_generator), distribution(mersenne_generator), distribution(mersenne_generator), _id);
	};



	void setX(double val) { coords[0] = val; };
	void setY(double val) { coords[1] = val; };
	void setZ(double val) { coords[2] = val; };
	void setid(int _id) { id = _id; };

	double getX() const { return coords[0]; };
	double getY() const { return coords[1]; };
	double getZ() const { return coords[2]; };
	int getid() const { return id; };
	std::vector<double> getCoords() const { return coords; };


	std::vector<Point> getNeighbors() const {
		return neighbors;
	};

	void addNeighbor(const Point& p) {
		neighbors.push_back(p);
	};

	void addNeighbors(const std::vector<Point>& to_add) {
		neighbors.insert(neighbors.end(), to_add.begin(), to_add.end());
	};

	void removeNeighbor(const Point& p) {
		const auto& where = std::find(neighbors.begin(), neighbors.end(), p);
		if (where != neighbors.end()) {
			neighbors.erase(where);
		};
		
	};

	void removeNeighbors(const std::vector<Point>& to_remove) {
		for (const Point& p : to_remove) {
			removeNeighbor(p);
		};
	};

	

	Point operator+(const Point& p) const {
		return Point(coords[0] + p.getX(), coords[1] + p.getY(), coords[2] + p.getZ());
	};
	Point operator-() {
		return Point(-coords[0], -coords[1], -coords[2]);
	};

	Point operator*(double factor) const {
		return Point(coords[0] * factor, coords[1] * factor, coords[2] * factor);
	};

	double operator*(Point p) const {
		return coords[0] * p.getX() + coords[1] * p.getY() + coords[2] * p.getZ();
	};

	Point operator=(const Point& rhs) {
		if (this == &rhs)
			return *this;
		this->setX(rhs.getX());
		this->setY(rhs.getY());
		this->setZ(rhs.getZ());
		this->setid(rhs.getid());
		return *this;
	};

	double Length() const {
		return sqrt((*this)*(*this));
	};

	Point  cross(const Point& rhs)const {
		std::vector<double> a = this->coords;
		std::vector<double> b = rhs.getCoords();
		return Point(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
	};

	bool operator==(const Point& rhs) const {
		return (this->getX() == rhs.getX() && this->getY() == rhs.getY() && this->getZ() == rhs.getZ());
	};

	bool operator<(const Point& rhs) const {
		return coords < rhs.coords;
	};

};

// Random
std::random_device Point::seed_generator;
// unsigned Point::seed = seed_generator();
unsigned Point::seed = 2313723809;

// 2313723809 - Vertices[2] is an interesting boundary point
std::uniform_real_distribution<double> Point::distribution(0, 1);
std::mt19937 Point::mersenne_generator(Point::seed);

// Initialization of the atomic static counter - counts points
// std::atomic<int> Point::s_id = 0;


Point operator-(Point p, Point q) {
	return p + (-q);
};

Point operator*(double factor, Point p) {
	return p * factor;
};


Point operator/(Point p, double divisor) {
	return p * (1 / divisor);
};

bool operator!=(Point p, Point q) {
	return !(p == q);
};






std::ostream& operator<<(std::ostream& out, Point p) {
	if (POINT_INDEX_ONLY) {
		out << "(" << p.getid() << ")";
	}
	else {
		out << p.getid() << ":(" << p.getX() << "," << p.getY() << "," << p.getZ() << ")"; // ":(" indeed
	};
	return out;
};



class Edge {
private:
	Point a, b;

public:
	Edge(Point _a, Point _b) : a(_a), b(_b) {};

	void setA(Point p) { a = p; };
	void setB(Point p) { b = p; };

	Point getA() const { return a; };
	Point getB() const { return b; };

	double Length() const { return sqrt((b - a)*(b - a)); };
};


std::ostream& operator<<(std::ostream& out, Edge l) {
	out << "[" << l.getA() << "," << l.getB() << "]";
	return out;
};



double SignedVolume(Point a, Point b, Point c, Point d) {
	return (1 / 6.0)*((b - a).cross(c - a)) * (d - a);
};


class Face {
private:
	Point a, b, c;
public:
	// Lexicographic order for hashmap (TODO: Improve? How does Joe do it?)
	Face(std::vector<Point> _v) {
		std::sort(_v.begin(), _v.end());
		a = _v[0];
		b = _v[1];
		c = _v[2];
	};
	Face(Point _a, Point _b, Point _c) {
		std::vector<Point> v = { _a,_b,_c };
		std::sort(v.begin(), v.end());
		a = v[0];
		b = v[1];
		c = v[2];
		// std::cout << a << v[0] << std::endl;
	};
	Face() : a(), b(), c() {}; // Default constructor for default HashNode

	void setA(Point p) { a = p; };
	void setB(Point p) { b = p; };
	void setC(Point p) { c = p; };

	Point getA() const { return a; };
	Point getB() const { return b; };
	Point getC() const { return c; };

	std::vector<Point> getPoints() const {
		std::vector<Point> v = { a,b,c };
		return v;
	};

	// Helper function for searching for neighbours. Possibly sub-optimal to do it this way.
	bool hasVertex(Point p) const {
		return(a == p || b == p || c == p);
	};

	// Find if a line intersects the face. Used for determining which swap operation to use. 
	// From https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
	bool intersectsLine(Point p, Point q) {
		return ((SignedVolume(p, a, b, c) * SignedVolume(q, a, b, c) < 0) && ((SignedVolume(p, q, a, b) * SignedVolume(p, q, b, c) > 0) &&
			(SignedVolume(p, q, b, c) * SignedVolume(p, q, c, a) > 0)));
	};


	bool isAdjacentTo(const Face& f) const {
		int count = 0;
		if (f.hasVertex(a)) { count++; };
		if (f.hasVertex(b)) { count++; };
		if (f.hasVertex(c)) { count++; };

		return (count >= 2);
	};



	double minEdgeLength() {
		return std::min((b - a).Length(), std::min((c - a).Length(), (c - b).Length()));
	};

	// https://math.stackexchange.com/questions/128991/how-to-calculate-area-of-3d-triangle
	double area() {
		Point AB = b - a;
		Point AC = c - a;

		return (1 / 2.0 * sqrt(pow(AB.getY()*AC.getZ() - AB.getZ()*AC.getY(), 2) + pow(AB.getZ()*AC.getX() - AB.getX()*AC.getZ(), 2) + pow(AB.getX()*AC.getY() - AB.getY()*AC.getX(), 2)));
	};


	bool operator==(const Face& rhs) const {
		return (this->getA() == rhs.getA() && this->getB() == rhs.getB() && this->getC() == rhs.getC());
	};
};


bool operator!=(Face f, Face g) {
	return !(f == g);
};

std::ostream& operator<<(std::ostream& out, Face f) {
	out << "[" << f.getA() << "," << f.getB() << "," << f.getC() << "]";
	return out;
};








//class Vertex {
//	x, y, z;
//	std::vector<Vertex*> Neighbours;np
//};












void getCofactor(double m[5][5], double cofactor[5][5], int p, int q, int n) {
	int i = 0, j = 0;

	// Loop over matrix m
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {

			if (row != p && col != q) {
				cofactor[i][j++] = m[row][col];
			};

		};
		// end of row
		if (j != 0) { j = 0; i++; };
	};
};

void writeMatrix(double m[5][5], int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << m[i][j] << " ";
		};
		std::cout << std::endl;
	};
};

double det(double m[5][5], int n) {
	double D = 0.0;

	if (n == 2) {
		return m[0][0] * m[1][1] - m[1][0] * m[0][1];
	}
	else {
		double cofactor[5][5];
		int sign = 1;
		// Iterate over the first row
		for (int i = 0; i < n;i++) {
			// Get cofactor matrix

			getCofactor(m, cofactor, 0, i, n);
			D += sign * m[0][i] * det(cofactor, n - 1);
			sign = -sign; // Sign changes each time
		};
		return D;
	};
};





// TODO: Bake in CCW into the code somehow? So that tetra are always CCW? While adding face, point d is such that it's CCW? etc..
double CCW(Point p0, Point p1, Point p2, Point p3) {
	double ccw[5][5] = { { p0.getX(), p0.getY(), p0.getZ(), 1 },
	{ p1.getX(), p1.getY(), p1.getZ(), 1 },
	{ p2.getX(), p2.getY(), p2.getZ(), 1 },
	{ p3.getX(), p3.getY(), p3.getZ(), 1 } };


	return det(ccw, 4);
}



double INCIRCLE(Point p0, Point p1, Point p2, Point p3, Point p4) {



	double m[5][5];

	double ccw = CCW(p0, p1, p2, p3);
	if (ccw < 0) {
		Point temp = p0;
		p0 = p1;
		p1 = temp;
	};


	if (VERBOSE2) {
		std::vector<Point> vertices = { p0,p1,p2,p3,p4 };
		std::cout << "Calling INCIRCLE on points: " << vertices << std::endl;
		std::cout << "CCW det: " << CCW(p0, p1, p2, p3) << std::endl;
	};





	// Weights. For power diagram would also include the radii
	double w0 = pow(p0.getX(), 2) + pow(p0.getY(), 2) + pow(p0.getZ(), 2);
	double w1 = pow(p1.getX(), 2) + pow(p1.getY(), 2) + pow(p1.getZ(), 2);
	double w2 = pow(p2.getX(), 2) + pow(p2.getY(), 2) + pow(p2.getZ(), 2);
	double w3 = pow(p3.getX(), 2) + pow(p3.getY(), 2) + pow(p3.getZ(), 2);
	double w4 = pow(p4.getX(), 2) + pow(p4.getY(), 2) + pow(p4.getZ(), 2);

	// Define INCIRCLE matrix
	m[0][0] = p0.getX();  m[0][1] = p0.getY(); m[0][2] = p0.getZ(); m[0][3] = w0; m[0][4] = 1;
	m[1][0] = p1.getX();  m[1][1] = p1.getY(); m[1][2] = p1.getZ(); m[1][3] = w1; m[1][4] = 1;
	m[2][0] = p2.getX();  m[2][1] = p2.getY(); m[2][2] = p2.getZ(); m[2][3] = w2; m[2][4] = 1;
	m[3][0] = p3.getX();  m[3][1] = p3.getY(); m[3][2] = p3.getZ(); m[3][3] = w3; m[3][4] = 1;
	m[4][0] = p4.getX();  m[4][1] = p4.getY(); m[4][2] = p4.getZ(); m[4][3] = w4; m[4][4] = 1;


	// Calculate the determinant of the INCRICLE matrix
	// TODO: Why do I need the minus? Is there a mistake somewhere?
	double det_result = -det(m, 5);




	if (VERBOSE2) {
		std::cout << "INCIRCLE det = " << det_result << std::endl;
	};

	return det_result;


};


double INCIRCLE(Face f, Point p, Point q) {

	return INCIRCLE(f.getA(), f.getB(), f.getC(), p, q);
};

double INCIRCLE(std::vector<Point> vertices) {
	return INCIRCLE(vertices[0], vertices[1], vertices[2], vertices[3], vertices[4]);
};


void copyMatrix(double A[5][5], double B[5][5], int n) {
	for (int i = 0;i < n;i++) {
		for (int j = 0; j < n; j++) {
			B[i][j] = A[i][j];
		};
	};
};




// Finds the index of element e in vector v
// Why not const std::vector<T>& v
// TODO: Lose this?
template <typename T>
int vectorIndex(const T& e, std::vector<T>& v) {
	return std::distance(v.begin(), (std::find(v.begin(), v.end(), e)));
}

template <typename T>
int vectorNegativeIndex(std::vector<T>& v) {
	return std::distance(v.begin(), std::find_if(v.begin(), v.end(), [](T i) { return i < 0; }));
}




class Tetrahedron {
private:
	Point a, b, c, d;
public:
	// TODO: Check for coplanarity - i.e. need cross product implementation
	Tetrahedron(Point _a, Point _b, Point _c, Point _d) : a(_a), b(_b), c(_c), d(_d) {};
	Tetrahedron(Face _f, Point _d) : a(_f.getA()), b(_f.getB()), c(_f.getC()), d(_d) {};
	Tetrahedron(std::vector<Point> _v) : a(_v[0]), b(_v[1]), c(_v[2]), d(_v[3]) {};

	void setA(Point p) { a = p; };
	void setB(Point p) { b = p; };
	void setC(Point p) { c = p; };
	void setD(Point p) { d = p; };

	Point getA() const { return a; };
	Point getB() const { return b; };
	Point getC() const { return c; };
	Point getD() const { return d; };

	std::vector<Point> getVertices() const { return std::vector<Point>({ a,b,c,d }); };

	// Pomocna funkce pro debilni id system TODO: Improve
	void setVerticeID(std::vector<int> id) {
		a.setid(id[0]);
		b.setid(id[1]);
		c.setid(id[2]);
		d.setid(id[3]);
	};

	Face getOpposingFace(int vertexIndex) const {
		std::vector<Point> vertices = { a,b,c,d };
		vertices.erase(vertices.begin() + vertexIndex);
		return Face(vertices);
	};

	std::vector<Face> getFaces() const {
		std::vector<Face> faces;
		faces.push_back(Face(a, b, c));
		faces.push_back(Face(a, b, d));
		faces.push_back(Face(a, c, d));
		faces.push_back(Face(b, c, d));
		return faces;
	};

	// To be able to use std::set<Tetrahedron>
	bool operator<(const Tetrahedron& rhs) const {
		return getVertices() < rhs.getVertices();
	};

	bool hasVertex(const Point& p) const {
		return ((a == p) || (b == p) || (c == p) || (d == p));
	};


	std::vector<double> getBarycentricCoordinates(Point p) const {
		double m[5][5] = { { a.getX(), b.getX(), c.getX(), d.getX() },
		{ a.getY(), b.getY(), c.getY(), d.getY() },
		{ a.getZ(), b.getZ(), c.getZ(), d.getZ() },
		{ 1,1,1,1 } };

		// std::cout << det(m, 4) << std::endl;

		std::vector<double> coordinates;
		double mCram[5][5];
		double det_m = det(m, 4);


		for (int i = 0; i < 4; i++) {
			copyMatrix(m, mCram, 4);
			mCram[0][i] = p.getX();
			mCram[1][i] = p.getY();
			mCram[2][i] = p.getZ();
			mCram[3][i] = 1;

			coordinates.push_back(det(mCram, 4) / det_m);
		}


		return coordinates;
	};

	Point getCentroid() const {
		return Point((a.getX() + b.getX() + c.getX() + d.getX()) / 4.0, (a.getY() + b.getY() + c.getY() + d.getY()) / 4.0, (a.getZ() + b.getZ() + c.getZ() + d.getZ()) / 4.0);
	};

	void enlargeYourTetrahedronBy6cm(double factor) {
		Point centroid = getCentroid();
		a = a + (a - centroid) * (factor - 1);
		b = b + (b - centroid) * (factor - 1);
		c = c + (c - centroid) * (factor - 1);
		d = d + (d - centroid) * (factor - 1);
	};

	// Checks if tetrahedron contains point p
	bool contains(Point p) const {
		std::vector<double> coords = getBarycentricCoordinates(p);
		return coords.begin() + vectorNegativeIndex(coords) == coords.end();
	};

	// Checks if tetrahedron contains all points in v
	bool contains(std::vector<Point> v) const {
		std::vector<bool> contain_vector;
		std::transform(v.begin(), v.end(), std::back_inserter(contain_vector), [&](Point p) { return contains(p); });
		return std::adjacent_find(contain_vector.begin(), contain_vector.end(), std::not_equal_to<bool>()) == contain_vector.end();
	};

	double volume() const {
		return(abs((1 / 6.0)*((b - a).cross(c - a)) * (d - a)));
	};


	Tetrahedron operator+(Point p) {
		return Tetrahedron(a + p, b + p, c + p, d + p);
	};

};

std::ostream& operator<<(std::ostream& out, Tetrahedron t) {
	out << "[" << t.getA() << "," << t.getB() << "," << t.getC() << "," << t.getD() << "]";
	return out;
};

Tetrahedron operator+(Point p, Tetrahedron tetra) {
	return tetra + p;
};









// Hash function for unordered map for Face

struct faceHash {
	size_t operator()(const Face& face) const {
		const unsigned int M = 10007;
		int n = 1000;
		int a = (face.getA()).getid();
		int b = (face.getB()).getid();
		int c = (face.getC()).getid();
		return (a*n*n + b * n + c) % M;
	}
};


// typedef std::pair <int, int> intpair;
class HashNode {
private:
	Face face;
	Point e;
	Point f;
	// HashNode* next; Not needed since unordered map resolves colisions by itself?

public:

	HashNode(const Face &_face, Point _e, Point _f) : face(_face), e(_e), f(_f) { if (f == Point(0, 0, 0)) { f = e; }; }; // TODO: Improve workaround?
																														  // Workaround : if e and f equal, there's no f. This should mean boundary face?
	HashNode() : face(), e(), f() {};


	Face getFace() const {
		return face;
	};

	void setFace(Face face) {
		HashNode::face = face;
	};

	std::vector<Point> getPoints() const {
		return std::vector<Point>({ e,f });
	};

	void setPoints(std::vector<Point> v) {
		e = v[0];
		f = v[1];
	};

	Point getPointe() const {
		return e;
	};

	Point getPointf() const {
		return f;
	};

	void setPointe(Point _e) {
		e = _e;
	};

	void setPointf(Point _f) {
		f = _f;
	};

	void changePoint(Point to_change, Point replacement) {

		if (VERBOSE3) {
			std::cout << "Changing face: " << face << "," << e << "," << f << std::endl;
			std::cout << "Changing " << to_change << " for " << replacement << std::endl;
		};

		if ((e != to_change) && (f != to_change)) {
			throw std::invalid_argument("changePoint: point to_change not found");
		};


		if (e == to_change) { e = replacement; };
		if (f == to_change) { f = replacement; };

	};

	Point getOppositePoint(const Point& p) const {

		if ((e != p) && (f != p)) {
			throw std::invalid_argument("changePoint: point p not found");
		};

		if (e == p) { return f; };
		if (f == p) { return e; };
	};

	bool isBoundary() const {
		return (e == f);
	};

	// Checks if a face is boundary, but isn't explicitly so because the bounding tetrahedron is still present in the tesellation
	bool isBoundaryHidden() const {
		return((e.getid() >= 0 && e.getid() <= 3) || (f.getid() >= 0 && f.getid() <= 3));
	};

	// HashNode* getNext() const {
	//	return next;
	// };



	// void setNext(HashNode* next) {
	//	HashNode::next = next;
	// };

};


std::ostream& operator<<(std::ostream& out, const HashNode& h) {
	out << h.getFace() << "," << h.getPointe() << "," << h.getPointf();
	return out;
};


class Tesselation {
public:
	std::vector<Point> Vertices;
	std::unordered_map<Face, HashNode, faceHash> FacesHT;

	Tesselation(std::vector<Point> _Vertices, std::unordered_map<Face, HashNode, faceHash> _FacesHT)
		: Vertices(_Vertices), FacesHT(_FacesHT) {};
	Tesselation() {
		std::vector<Point> Vertices;
		std::unordered_map<Face, HashNode, faceHash> FacesHT;
	};

	void addFace(const Face& face, const Point& e = Point(0, 0, 0), const Point& f = Point(0, 0, 0)) {
		if (VERBOSE3) {
			std::cout << "Adding face " << face << e << f << std::endl;
		};
		
		HashNode hnode(face, e, f);
		FacesHT.insert(std::make_pair(face, hnode));
		
		/*std::vector<Point> v = face.getPoints();
		v[0].addNeighbor(v[1]);
		v[0].addNeighbor(v[2]);
		v[1].addNeighbor(v[0]);
		v[1].addNeighbor(v[2]);
		v[2].addNeighbor(v[0]);
		v[2].addNeighbor(v[1]);*/
	};

	void removeFace(Face face) {
		if (VERBOSE3) {
			std::cout << "Removing face " << FacesHT.at(face) << std::endl;
		};
		FacesHT.erase(face);

		
		/*std::vector<Point> v = face.getPoints();
		v[0].removeNeighbor(v[1]);
		v[0].removeNeighbor(v[2]);
		v[1].removeNeighbor(v[0]);
		v[1].removeNeighbor(v[2]);
		v[2].removeNeighbor(v[0]);
		v[2].removeNeighbor(v[1]);*/
	};

	// Searches through all faces and finds those containing point p, thus finding its neighbours
	// TODO: Possibly slow, optimize (e.g. actually save this information)
	std::set<Point> findNeighborsStupid(Point p) {
		std::set<Point> Neighbors;

		// Go through all faces
		for (const auto& hnode_pair : FacesHT) {
			// If it contains the vertex
			if (hnode_pair.first.hasVertex(p)) {
				std::vector<Point> face_vertices = hnode_pair.first.getPoints();
				// Add all the vertices in the face (the point itself will be removed)
				std::copy(face_vertices.begin(), face_vertices.end(), std::inserter(Neighbors, Neighbors.end()));
			};
		};
		//Remove the point
		Neighbors.erase(p);

		return Neighbors;
	};

	void addBoundingTetrahedron(Tetrahedron tetra) {

		// Add the points at the beginning of Vertices
		std::vector<Point> bounding_vertices = tetra.getVertices();
		Vertices.insert(Vertices.begin(), bounding_vertices.begin(), bounding_vertices.end());

		for (int i = 0; i <= 3; i++) {
			Face face = tetra.getOpposingFace(i);
			addFace(face, bounding_vertices[i]); // TODO: Searching for the face - inefficient
		};
	};


	// TODO NOHIC: Polymorphism for bounding tetrahedron?

	// Returns approximately 10 * the tetrahedron that bounds all of Vertices
	// TODO: Make it non-heuristic
	// 1: Tetrahedron shape - i.e. regular, centered
	// 2: Enlarging algorithm - more accurate, with some proper reasoning behind it
	//void createBoundingTetrahedron(const std::vector<Point>& points_to_bound, const std::vector<int>& ids = { 0,1,2,3 }) {
	//	Point centroid = std::accumulate(points_to_bound.begin(), points_to_bound.end(), Point(0, 0, 0)) / points_to_bound.size();
	//	// std::cout << Tetrahedron(Point(-1 - 1, -0.5), Point(1, -1, -0.5), Point(0, 1, -0.5), Point(0, 0, 1)).getCentroid() << std::endl;
	//	Tetrahedron bounding_tetra(Point(-1 - 1, -0.5) + centroid, Point(1, -1, -0.5) + centroid, Point(0, 1, -0.5) + centroid, Point(0, 0, 1) + centroid);
	//	if (VERBOSE) {
	//		std::cout << "::: createBoundingTetrahedron()" << std::endl;
	//		std::cout << "Beginning with: " << bounding_tetra << std::endl;
	//		std::cout << "Centroid: " << centroid << std::endl;
	//		std::cout << "tetra centroid: " << bounding_tetra.getCentroid() << std::endl;
	//	};
	//	while (!bounding_tetra.contains(points_to_bound)) {
	//		bounding_tetra.enlargeYourTetrahedronBy6cm(2);
	//		if (VERBOSE) {
	//			std::cout << "Inflated: " << bounding_tetra << std::endl;
	//		};
	//	};
	//	bounding_tetra.enlargeYourTetrahedronBy6cm(10000);
	//	if (VERBOSE) {
	//		std::cout << "Final inflation: " << bounding_tetra << std::endl;
	//	};
	//	// Set id in a stupid way
	//	bounding_tetra.setVerticeID(ids);


	//	// Then add the tetrahedron
	//	addBoundingTetrahedron(bounding_tetra);
	//};

	// Based on Edelsbrunner & Shah (1996)
	Tetrahedron createBoundingTetrahedron() {
		double xi(3 * pow(10, 5));
		
		Point a(0, 0, xi,-5);
		Point b(0, xi, -xi,-4);
		Point c(xi, -xi, -xi,-3);
		Point d(-xi, -xi, -xi,-2);

	
		Tetrahedron bounding_tetra(a, b, c, d);
		addBoundingTetrahedron(bounding_tetra);

		return bounding_tetra;
	};

	Tetrahedron getBoundingTetrahedron() const {
		return Tetrahedron(Vertices[0], Vertices[1], Vertices[2], Vertices[3]);
	};




	void deleteBoundingTetrahedron() {
		/*Faces.erase(
		std::remove_if(Faces.begin(), Faces.end(),
		[this](Face f) {return (f.hasVertex(Vertices[0]) || f.hasVertex(Vertices[1]) || f.hasVertex(Vertices[2]) || f.hasVertex(Vertices[3])); }) ,
		Faces.end());
		*/

		
		std::vector<Point> bp = getBoundingTetrahedron().getVertices(); // Points of bounding tetrahedron

		std::vector<Face> faces_to_delete;

		for (const auto& hash_pair : FacesHT) {
			Face f = hash_pair.first;
			// If the face is adjacent to the bounding tetrahedron's vertices, delete
			if (f.hasVertex(bp[0]) || f.hasVertex(bp[1]) || f.hasVertex(bp[2]) || f.hasVertex(bp[3])) {
				faces_to_delete.push_back(f);
				continue;
			};

			// If the face has bounding tetrahedron's vertices as an opposite point, it should actaully be a bounding face
			if (std::find(bp.begin(), bp.end(), FacesHT.at(f).getPointe()) != bp.end()) {
				FacesHT.at(f).setPointe(FacesHT.at(f).getPointf());
				continue;
			}
			else
				if (std::find(bp.begin(), bp.end(), FacesHT.at(f).getPointf()) != bp.end()) {
					FacesHT.at(f).setPointf(FacesHT.at(f).getPointe());
					continue;
				};

		};

		for (const Face& f : faces_to_delete) {
			removeFace(f);
		};

		for (Point p : Vertices) {
			p.removeNeighbors(bp);
		};


		Vertices.erase(Vertices.begin());
		Vertices.erase(Vertices.begin());
		Vertices.erase(Vertices.begin());
		Vertices.erase(Vertices.begin());
	};

	void addPointToTetrahedron(Point p, Tetrahedron t) {
		std::vector<Point> v = t.getVertices();
		std::sort(v.begin(), v.end()); // std::set_difference requires sorted input

									   // Alter the existing faces of the tetrahedron: they now have a different opposing point
		for (int i = 0; i <= 3; i++) {
			std::vector<Point> v_copy = v; // a copy of v to erase from
			Point opp_point = v_copy[i]; // the point that is not deleted is the opposite to the face
			v_copy.erase(v_copy.begin() + i);
			std::vector<Point> opp_points = FacesHT.at(Face(v_copy)).getPoints();
			// TODO: Improve handling of boundary faces
			if (opp_points[0] == opp_points[1]) {
				opp_points[0] = p;
				opp_points[1] = p;
			}
			else {
				opp_points[vectorIndex(opp_point, opp_points)] = p;
			};
			FacesHT.at(Face(v_copy)).setPoints(opp_points);
		};

		// Create 6 new faces
		// Choose all pairs (a,b) to add face (a,b,p) with opposite points (c,d)
		for (int i = 0; i <= 3; i++) {
			for (int j = i + 1; j <= 3; j++) {
				// v[i] and v[j] are the vertices forming the face
				std::vector<Point> to_add;
				to_add.push_back(v[i]);
				to_add.push_back(v[j]);
				// the other two points form two tetrahedrons with the face
				std::vector<Point> opp_pts;
				std::set_difference(v.begin(), v.end(), to_add.begin(), to_add.end(), std::inserter(opp_pts, opp_pts.end()));
				addFace(Face(to_add[0], to_add[1], p), opp_pts[0], opp_pts[1]);

			};
		};


	};


	void writeFaces() const {
		std::cout << "===Faces===" << std::endl;
		for (const auto& hp: FacesHT) {
		std::cout << hp.second << std::endl;
		};
		std::cout << std::endl;
	};
	

	// Performs INCIRCLE test on the Face f
	bool isLocallyOptimal(Face f) const {
		HashNode hnode = FacesHT.at(f);
		double incircle_result = INCIRCLE(hnode.getFace(), hnode.getPointe(), hnode.getPointf());
		// TODO: -0 or 0 checking (boundary faces)
		return incircle_result >= 0;
	};


	// Checks whether the tesselation is a Delaunay tesselation by testing INCIRCLE on all faces
	bool isDelaunay() const {
		if (VERBOSE) {
			std::cout << "::isDelaunay" << std::endl;
		};
		bool result = true;

		for (const auto& hnode_pair : FacesHT) {
			// std::cout << "check loop: " << i << " " << Faces.size() << Faces[i] << std::endl;
			if (!isLocallyOptimal(hnode_pair.first)) {
				result = false;
				if (VERBOSE) {
					std::cout << "Not locally optimal face: " << hnode_pair.second << std::endl;
				};
			};
		};
		if (VERBOSE) {
			std::cout << "isDelaunay result: " << result << std::endl;
		};

		return result;
	};

	// Swap how-to
	/*	Swap [0, 1, 2], 3, 4

	Add:
	[3, 4, 0], 1, 2
	[3, 4, 1], 0, 2
	[3, 4, 2], 1, 2

	Change
	[0, 1, 3] 2 -> 4
	[1, 2, 3] 0 -> 4
	[0, 2, 3] 1 -> 4

	[0, 1, 4] 2 -> 3
	[1, 2, 4] 0 -> 3
	[0, 2, 4] 1 -> 3

	Remove:
	0, 1, 2
	*/
	void Swap23(Face face_swap) {
		std::vector<Point> pts = face_swap.getPoints();
		Point e = FacesHT.at(face_swap).getPointe();
		Point f = FacesHT.at(face_swap).getPointf();


		if (VERBOSE) {
			std::cout << "::: Swap 2-3" << std::endl;
			std::cout << "Swapping the tetrahedra joined at " << face_swap << " with endpoints " << e << " and " << f << std::endl;
		};

		// Add 3 new inner faces
		addFace(Face(e, f, pts[0]), pts[1], pts[2]);
		addFace(Face(e, f, pts[1]), pts[0], pts[2]);
		addFace(Face(e, f, pts[2]), pts[0], pts[1]);

		// Change 6 outer faces' opposite points
		FacesHT.at(Face(pts[0], pts[1], e)).changePoint(pts[2], f);
		FacesHT.at(Face(pts[1], pts[2], e)).changePoint(pts[0], f);
		FacesHT.at(Face(pts[0], pts[2], e)).changePoint(pts[1], f);

		FacesHT.at(Face(pts[0], pts[1], f)).changePoint(pts[2], e);
		FacesHT.at(Face(pts[1], pts[2], f)).changePoint(pts[0], e);
		FacesHT.at(Face(pts[0], pts[2], f)).changePoint(pts[1], e);

		// Remove the current inner face
		removeFace(face_swap);


	};

	// Points b,c are the new opposite points
	void swap32(Point a, Point b, Point c) {
		if (VERBOSE) {
			std::cout << "::: Swap 3-2" << std::endl;
			std::cout << "Pivot face: " << Face(a, b, c) << std::endl;
		};



		Face face_swap = Face(a, b, c);
		Point d = FacesHT.at(face_swap).getPointe();
		Point e = FacesHT.at(face_swap).getPointf();


		addFace(Face(a, d, e), b, c);

		removeFace(Face(a, b, c));
		removeFace(Face(b, c, d));
		removeFace(Face(b, c, e));

		FacesHT.at(Face(a, b, d)).changePoint(c, e);
		FacesHT.at(Face(a, c, d)).changePoint(b, e);

		FacesHT.at(Face(a, b, e)).changePoint(c, d);
		FacesHT.at(Face(a, c, e)).changePoint(b, d);


		FacesHT.at(Face(c, d, e)).changePoint(b, a);
		FacesHT.at(Face(b, d, e)).changePoint(c, a);




	};

	// Finds first nonOptimal face -- may not return anything
	/*std::vector<Face> findNonOptimalFaces() const {
	std::vector<Face> non_optimal_faces;

	for (size_t i = 0; i < Faces.size(); i++) {
	HashNode hnode = FacesHT.at(Faces[i]);
	double incircle_result = INCIRCLE(hnode.getFace(), hnode.getPointe(), hnode.getPointf());
	if (incircle_result < 0) {
	non_optimal_faces.push_back(Faces[i]);
	};
	};

	return(non_optimal_faces);

	};

	*/

	//void balanceStupid() {
	//	if (VERBOSE) {
	//		std::cout << "::balanceStupid()" << std::endl;
	//	};

	//	while (!isDelaunay()) {
	//		std::vector<Face> non_optimal_faces = findNonOptimalFaces();
	//		if (VERBOSE) {
	//			std::cout << "Non-optimal faces" << non_optimal_faces << std::endl;
	//		};

	//		for (const Face& f_no: non_optimal_faces) {
	//			// If the face is not in the triangulation anymore, remove it and continue
	//			if (FacesHT.count(f_no) == 0) {
	//				continue;
	//			};


	//			std::vector<Point> pts = f_no.getPoints();
	//			Point e = FacesHT[f_no].getPointe();
	//			Point f = FacesHT[f_no].getPointf();

	//			if (VERBOSE) {
	//				std::cout << "Non-optimal face" << FacesHT[f_no] << std::endl;
	//			};

	//			if (VERBOSE) {
	//				std::cout << "Intersection checks: " << Face(pts[2], e, f).intersectsLine(pts[0], pts[1]) << " " << Face(pts[1], e, f).intersectsLine(pts[0], pts[2]) << " " << Face(pts[0], e, f).intersectsLine(pts[1], pts[2]) << std::endl;
	//			};

	//			if (Face(pts[2], e, f).intersectsLine(pts[0], pts[1])) {
	//				if (FacesHT.count(Face(e, f, pts[0])) == 0 || FacesHT.count(Face(e, f, pts[1])) == 0) {
	//					if (VERBOSE) {
	//						std::cout << "NON-TRANSFORMABLE FACE, CONTINUE" << std::endl;
	//					};
	//					continue;
	//				};

	//				swap32(pts[2], pts[0], pts[1]);
	//			}
	//			else if (Face(pts[1], e, f).intersectsLine(pts[0], pts[2])) {
	//				if (FacesHT.count(Face(e, f, pts[0])) == 0 || FacesHT.count(Face(e, f, pts[2])) == 0) {
	//					if (VERBOSE) {
	//						std::cout << "NON-TRANSFORMABLE FACE, CONTINUE" << std::endl;
	//					};
	//					continue;
	//				};

	//				swap32(pts[1], pts[0], pts[2]);
	//			}
	//			else if (Face(pts[0], e, f).intersectsLine(pts[1], pts[2])) {
	//				if (FacesHT.count(Face(e, f, pts[1])) == 0 || FacesHT.count(Face(e, f, pts[2])) == 0) {
	//					if (VERBOSE) {
	//						std::cout << "NON-TRANSFORMABLE FACE, CONTINUE" << std::endl;
	//					};
	//					continue;
	//				};

	//				swap32(pts[0], pts[1], pts[2]);
	//			}
	//			else {

	//				Swap23(f_no);

	//			};
	//		};
	//	};
	//};


	void balance(const std::vector<Face>& initial_faces) {
		std::queue<Face> FaceQueue;

		// Add initial faces to queue
		for (const Face& f : initial_faces) {
			FaceQueue.push(f);
		};

		if (VERBOSE) {
			std::cout << "::: balance" << std::endl;
		};


		while (!FaceQueue.empty()) {

			Face current_face = FaceQueue.front();

			// If the face is not in the triangulation anymore, remove it and continue
			if (FacesHT.count(current_face) == 0) {
				FaceQueue.pop();
				continue;
			};

			if (VERBOSE4) {
				if (FaceQueue.size() > 30) {
					std::cout << FaceQueue.size() << " ";
				};
			};


			if (!isLocallyOptimal(current_face)) {
				if (VERBOSE) {
					std::cout << "Locally non-optimal face: " << FacesHT.at(current_face) << std::endl;
				};


				// TODO: Write swaps more elegantly e.g. w.r.t. arguments

				std::vector<Point> pts = current_face.getPoints();
				Point d = FacesHT.at(current_face).getPointe();
				Point e = FacesHT.at(current_face).getPointf();

				if (VERBOSE) {
					std::cout << "Intersection checks: " << Face(pts[2], d, e).intersectsLine(pts[0], pts[1]) << " " << Face(pts[1], d, e).intersectsLine(pts[0], pts[2]) << " " << Face(pts[0], d, e).intersectsLine(pts[1], pts[2]) << std::endl;
				};

				if (Face(pts[2], d, e).intersectsLine(pts[0], pts[1])) {

					if (FacesHT.count(Face(d, e, pts[0])) == 0 || FacesHT.count(Face(d, e, pts[1])) == 0) {
						if (VERBOSE) {
							std::cout << "NON-TRANSFORMABLE FACE, CONTINUE" << std::endl;
						};
						FaceQueue.pop();
						continue;
					};

					Point a = pts[2];
					Point b = pts[0];
					Point c = pts[1];

					FaceQueue.push(Face(a, b, e));
					FaceQueue.push(Face(a, c, e));
					FaceQueue.push(Face(a, b, d));
					FaceQueue.push(Face(a, c, d));
					FaceQueue.push(Face(b, d, e));
					FaceQueue.push(Face(c, d, e));



					swap32(pts[2], pts[0], pts[1]);
				}
				else if (Face(pts[1], d, e).intersectsLine(pts[0], pts[2])) {

					if (FacesHT.count(Face(d, e, pts[0])) == 0 || FacesHT.count(Face(d, e, pts[2])) == 0) {
						if (VERBOSE) {
							std::cout << "NON-TRANSFORMABLE FACE, CONTINUE" << std::endl;
						};
						FaceQueue.pop();
						continue;
					};


					swap32(pts[1], pts[0], pts[2]);

					Point a = pts[1];
					Point b = pts[0];
					Point c = pts[2];

					FaceQueue.push(Face(a, b, e));
					FaceQueue.push(Face(a, c, e));
					FaceQueue.push(Face(a, b, d));
					FaceQueue.push(Face(a, c, d));
					FaceQueue.push(Face(b, d, e));
					FaceQueue.push(Face(c, d, e));

				}
				else if (Face(pts[0], d, e).intersectsLine(pts[1], pts[2])) {

					if (FacesHT.count(Face(d, e, pts[1])) == 0 || FacesHT.count(Face(d, e, pts[2])) == 0) {
						if (VERBOSE) {
							std::cout << "NON-TRANSFORMABLE FACE, CONTINUE" << std::endl;
						};
						FaceQueue.pop();
						continue;
					};


					swap32(pts[0], pts[1], pts[2]);

					Point a = pts[0];
					Point b = pts[1];
					Point c = pts[2];

					FaceQueue.push(Face(a, b, e));
					FaceQueue.push(Face(a, c, e));
					FaceQueue.push(Face(a, b, d));
					FaceQueue.push(Face(a, c, d));
					FaceQueue.push(Face(b, d, e));
					FaceQueue.push(Face(c, d, e));
				}
				else {
					if (VERBOSE) {
						std::cout << "Pushed three faces into the queue. Queue size: " << FaceQueue.size() << std::endl;
					};

					// Swap the tetra
					Swap23(current_face);

					Point a = pts[0];
					Point b = pts[1];
					Point c = pts[2];

					FaceQueue.push(Face(a, b, e));
					FaceQueue.push(Face(a, c, e));
					FaceQueue.push(Face(a, b, d));
					FaceQueue.push(Face(a, c, d));
					FaceQueue.push(Face(b, c, e));
					FaceQueue.push(Face(b, c, d));
				};

			};

			// Whether it was swapped or not, pop the face out of the queue
			FaceQueue.pop();
		};


	};

	//

	// TODO: random tetra initialization
	// TODO: Actually make it stochastic.. 
	Tetrahedron stochasticWalk(const Point& p) {
		if (VERBOSE) {
			std::cout << "::: stochasticWalk() " << std::endl;
			std::cout << "Searching for " << p << std::endl;
		};
		// Choose a random tetrahedron
		// TODO: Improve this
		Face start_face = FacesHT.begin()->second.getFace();
		Tetrahedron current_tetra(start_face, FacesHT.at(start_face).getPointe());



		if (VERBOSE) {
			std::cout << "Starting in " << current_tetra << std::endl;
		};

		// Calculate coordinates
		std::vector<double> coords = current_tetra.getBarycentricCoordinates(p);
		int negInd = vectorNegativeIndex(coords);
		while (coords.begin() + negInd != coords.end()) {

			Point initial_point = current_tetra.getVertices()[negInd];
			// Go to the tetrahedron in the direction of the point
			Face opposing_face = current_tetra.getOpposingFace(negInd);
			HashNode opposing_hnode = FacesHT.at(opposing_face);
			// both_points = { initial_point, opposite_point }
			std::vector<Point> both_points = { opposing_hnode.getPointe(), opposing_hnode.getPointf() };
			Point opposing_point = both_points[1 - vectorIndex(initial_point, both_points)];
			current_tetra = Tetrahedron(opposing_face, opposing_point);
			// Calculate coordinates w.r.t. the new tetrahedron
			coords = current_tetra.getBarycentricCoordinates(p);
			negInd = vectorNegativeIndex(coords);
			if (VERBOSE) {
				std::cout << "Walked into " << current_tetra << std::endl;
				// std::cout << both_points << std::endl;
				// std::cout << coords << std::endl;
			};

			// If the tetrahedron is infinite, terminate the walk
			
		};
		if (VERBOSE) {
			std::cout << "Found in " << current_tetra << std::endl;
		};
		return current_tetra;
	};

	// For point deletion - TODO: Bake this into incremental?
	std::vector<Face> findBoundaryFaces() const {
		std::vector<Face> boundary_faces;
		for (const auto& hash_pair : FacesHT) {
			if (hash_pair.second.isBoundary()) {
				boundary_faces.push_back(hash_pair.first);
			};
		};
		return boundary_faces;
	};

	// For point deletion
	std::vector<Face> findFacesContainingStupid(const Point& p) {
		// TODO: This is o(n), improve
		std::vector<Face> faces_contain;

		for (const auto& hash_pair : FacesHT) {
			if (hash_pair.first.hasVertex(p)) {
				faces_contain.push_back(hash_pair.first);
			};
		};

		return faces_contain;
	};

	// TODO: Add option to remember boundary faces?
	/*void addPoints(const std::vector<Point>& points_to_add) {
		int npts_current = Vertices.size();
		int npts_final = npts_current + points_to_add.size();

		Vertices.insert(Vertices.end(), points_to_add.begin(), points_to_add.end());


		for (int current_point = npts_current; current_point < npts_final; current_point++) {
			std::cout << std::endl;
			std::cout << "Adding " << current_point - 3 << "-th point " << std::endl;
			Tetrahedron containing_tetra = stochasticWalk(Vertices[current_point]);
			addPointToTetrahedron(Vertices[current_point], containing_tetra);
			balance(containing_tetra.getFaces());
		};
	};*/

	void addPoint(const Point p) {
		Vertices.push_back(p);
		Tetrahedron containing_tetra = stochasticWalk(p);
		addPointToTetrahedron(p, containing_tetra);
		balance(containing_tetra.getFaces());
	};


	void sewIn(Tesselation& DTlink, const Point& q) {
		
		// Removing unnecessary simplices
		// Approach idea: need to remove all simplices that do not have the empty sphere property
		// For this data structure, go through all faces and check both of their opposite points
		// Boundary face - only one incircle needs to be done
		// Non-boundary - both sides do not have ESP => remove the face
		// Non-boundary - one side does not have ESP => should be boundary, change opposite point
		// 
		for (const auto& hnp : DTlink.FacesHT) {
			Face f = hnp.first;
			HashNode hn = hnp.second;

			// Only check one side for a boundary face
			if (hn.isBoundary()) {
				if (INCIRCLE(f, hn.getPointe(), q) >= 0) {
					DTlink.removeFace(f);
				};
			}
			else
			{
				if (INCIRCLE(f, hn.getPointe(), q) >= 0) {

					if (INCIRCLE(f, hn.getPointf(), q) >= 0) {
						DTlink.removeFace(f);  // If both sides fail incircle, the face should be removed
					};
					hn.setPointe(hn.getPointf()); // If only one side fails, the face is a part of boundary of DTlink
				};
			};

		};

		// Sewing in
		for (const auto& hnp : DTlink.FacesHT) {
			Face f = hnp.first;
			HashNode hn = hnp.second;


			if (FacesHT.count(f) == 0 ) { // Non-boundary

			}
		};
		
	};


	void sewInOld(Tesselation DTlink, const Point& q) {   // Pass by..?
		std::vector<Face> bounding_faces = DTlink.findBoundaryFaces();
		std::queue<Face> BoundFaceQueue;
		int pomcounter = 0;

		// Add initial bnd(DTlink) faces to queue
		for (const Face& f : bounding_faces) {
			BoundFaceQueue.push(f);
		};

		// Find true boundary by deleting fake boundary faces
		while (!BoundFaceQueue.empty()) {
			if (pomcounter > 50) { break; };


			Face f = BoundFaceQueue.front();

			std::cout << "Queue size: " << BoundFaceQueue.size() << std::endl;
			std::cout << "Current face: " << DTlink.FacesHT.at(f) << std::endl;

			// Not in DT or (in DT but doesn't have q as opposite) => it's a faake! => Need to add faces "below" it
			if (
				((FacesHT.count(f) == 0) && (!f.intersectsLine(DTlink.FacesHT.at(f).getPointe(), q))) 
				|| 
				((FacesHT.count(f) == 1) && (!isIn(FacesHT.at(f).getPoints(), q))) 
				) {
				


				std::cout << "Fake boundary: " << f << std::endl;
				if ((FacesHT.count(f) == 0) && (!f.intersectsLine(DTlink.FacesHT.at(f).getPointe(), q)))
				{
					std::cout << "Not in FacesHT and not boundary" << std::endl;
				}
				else
				{
					std::cout << "Not connected to " << q << ". Hashnode: " << FacesHT.at(f) << std::endl;
				};


				Point p = DTlink.FacesHT.at(f).getPointe();
				Face f1(f.getA(), f.getB(), p);
				Face f2(f.getA(), f.getC(), p);
				Face f3(f.getB(), f.getC(), p);

				// Setting up a faces vector to add to queue - add only those faces that exist in DTlink (e.g. haven't been removed)
				std::vector<Face> faces;
				if (DTlink.FacesHT.count(f1) == 1) { faces.push_back(f1); };
				if (DTlink.FacesHT.count(f2) == 1) { faces.push_back(f2); };
				if (DTlink.FacesHT.count(f3) == 1) { faces.push_back(f3); };

				std::cout << "faces length " << faces.size() << std::endl;


				for (const Face& f1 : faces) {
					std::vector<Point> f1_opp = DTlink.FacesHT.at(f1).getPoints();

					if (!DTlink.FacesHT.at(f1).isBoundary()) {
						BoundFaceQueue.push(f1);
						std::cout << f1 << "added" << std::endl;
					};

					// Change the opposite point to make f1 boundary
					if (f.hasVertex(f1_opp[0])) {  // If this isn't the case, then f1_opp[1] has to be there
						DTlink.FacesHT.at(f1).changePoint(f1_opp[0], f1_opp[1]);
					}
					else
					{
						DTlink.FacesHT.at(f1).changePoint(f1_opp[1], f1_opp[0]);
					};


				DTlink.removeFace(f);

				};
			};



			pomcounter++;
			BoundFaceQueue.pop();
		};


		for (const auto& f : DTlink.FacesHT) {
			std::cout << f.second << std::endl;
		};



		for (const auto& f : DTlink.FacesHT) {
			if (f.second.isBoundary() && (FacesHT.count(f.first) == 1)) {
				std::cout << "Reconnect: " << FacesHT.at(f.first) << " reconnect " << q << " to " << f.second.getPointe() << std::endl;
				FacesHT.at(f.first).changePoint(q, f.second.getPointe());

			}
			else {
				std::cout << "Add face " << f.second << std::endl;
				addFace(f.first, f.second.getPointe(), f.second.getPointf());
			};
		};

	};






	// Returns the four neighbors of a tetrahedron
	// For boundary it returns the tetrahedron itself
	std::vector<Tetrahedron> findTetraNeighbors(const Tetrahedron& t) {
		std::vector<Point> v = t.getVertices();
		std::vector<Tetrahedron> neighbors;



		for (int i = 0; i < 4; i++) {
			Face f(v[(i + 1) % 4], v[(i + 2) % 4], v[(i + 3) % 4]);   // Obtain the other three elements of v
			Point opp_pt = FacesHT.at(f).getOppositePoint(v[i]);
			neighbors.push_back(Tetrahedron(f, opp_pt));
		};


		return neighbors;

	};



	void removePoint(const Point& q) {
		std::unordered_map<Face, HashNode, faceHash> Star;
		std::unordered_map<Face, HashNode, faceHash> Link;
		std::set<Point> Neighbors;
		std::queue<Tetrahedron> TetraQueue;
		std::set<Tetrahedron> ProcessedTetra;


		
		////// Construct Star(q) and Link(q)
		
		// Find first face adjacent to q
		Tetrahedron t = stochasticWalk(q);
		TetraQueue.push(t);

		while (!TetraQueue.empty()) {
			Tetrahedron t = TetraQueue.front();
			TetraQueue.pop();
			
			// Skip if we already processed the tetrahedron
			if (ProcessedTetra.count(t) == 1) {
				continue;
			};
					
			// Process the tetrahedron
			std::vector<Face> faces = t.getFaces();

			// Add neighbors
			std::vector<Point> t_neigh = t.getVertices();
			Neighbors.insert(t_neigh.begin(), t_neigh.end());

			// Add faces
			for (Face f : faces) {						// Add all faces...
				HashNode hn = FacesHT.at(f);
				
				Star.insert(std::make_pair(f, hn));		// ..to Star(q)
				
				if (!f.hasVertex(q)) {
					Link.insert(std::make_pair(f, hn)); // ..and to Link(q) if the face is not adjacent to q
				};
			};
			ProcessedTetra.insert(t);
			
			
			// Find neighboring tetrahedrons and add the ones incident to q
			std::vector<Tetrahedron> tetra_neighbors = findTetraNeighbors(t);			
			for (const Tetrahedron& t : tetra_neighbors) {
				if (t.hasVertex(q)) {
					TetraQueue.push(t);
				};
			};
		};


		for (auto hp : Link) {
			std::cout << hp.second << std::endl;
		};


		// Finding valid ears - a 2-ear is valid if it is convex from outside of Star(q), 3-ear is just three 2-ears
		// Ear - essentially any pair/triplet of adjacent faces in Link(q)? 

		// Find an ear
		Face f = Link.begin()->first;
		std::pair<Face, Face> ear;

		for (auto hp : Link) {
			if (hp.first.isAdjacentTo(f) & (hp.first!=f)) {
				ear = std::make_pair(f, hp.first);
				break;
			};
		};

		






	};


	//void deletePointOld(Point q) {
	//	if (VERBOSE5) {
	//		std::cout << "::deletePoint" << std::endl << "Deleting point " << q << std::endl;
	//	}

	//	// 1 Getting neighbours - TODO: set -> vector stupid
	//	std::set<Point> neighbors_set = findNeighborsStupid(q);
	//	std::cout << "True neighbors " << neighbors_set << std::endl;
	//	// Remove bounding points
	//	neighbors_set.erase(Vertices[0]);
	//	neighbors_set.erase(Vertices[1]);
	//	neighbors_set.erase(Vertices[2]);
	//	neighbors_set.erase(Vertices[3]);

	//	// Copy to a vector
	//	std::vector<Point> neighbors;
	//	neighbors.insert(neighbors.end(), neighbors_set.begin(), neighbors_set.end());

	//	if (VERBOSE5) {
	//		std::cout << "Neighbors: " << neighbors.size() << neighbors << std::endl;
	//	};

	//	// Make a cavity: Deleting the point and faces adjacent to it
	//	Vertices.erase(std::find(Vertices.begin(), Vertices.end(), q));
	//	std::vector<Face> adjacent_faces = findFacesContainingStupid(q);
	//	std::cout << "Adjacent faces " << std::endl;
	//	for (const Face& f : adjacent_faces) {
	//		std::cout << FacesHT.at(f) << FacesHT.at(f).isBoundaryHidden() << std::endl;
	//		removeFace(f);
	//	};

	//	// 2 Construct DTlink
	//	Tesselation DTlink;
	//	DTlink.addBoundingTetrahedron(getBoundingTetrahedron());
	//	std::cout << DTlink.Vertices << std::endl;
	//	std::cout << "Contain check: " << DTlink.getBoundingTetrahedron().contains(neighbors) << std::endl;
	//	DTlink.addPoints(neighbors);
	//	DTlink.deleteBoundingTetrahedron();




	//	if (VERBOSE5) {
	//		std::cout << "DTlink Delaunay check: " << DTlink.isDelaunay() << std::endl;
	//		std::cout << "DTlink no. of faces: " << DTlink.FacesHT.size() << std::endl;
	//		std::cout << "DTlink vertices: " << DTlink.Vertices << std::endl;
	//		std::cout << "DTlink faces: " << std::endl;
	//		for (const auto& hn : DTlink.FacesHT) {
	//			std::cout << hn.second << std::endl;
	//		};
	//	};

	//	// 3 Sew DTlink in DT
	//	sewInOld(DTlink, q);
	//};
};










// Template od Nohice
template <typename T>
using matrix = std::vector<std::vector<T>>;
// TODO: Re-implement matrices?






// Checks collinearity and cosphericality issues for five points.
void checkPoints(const std::vector<Point>& pts) {
	std::cout << pts << std::endl;

	Point a = pts[0];
	Point b = pts[1];
	Point c = pts[2];
	Point d = pts[3];
	Point e = pts[4];

	std::cout << Face(a, b, c).area() << std::endl;
	std::cout << Face(a, b, d).area() << std::endl;
	std::cout << Face(a, b, e).area() << std::endl;
	std::cout << Face(a, c, d).area() << std::endl;
	std::cout << Face(a, c, e).area() << std::endl;
	std::cout << Face(a, d, e).area() << std::endl;
	std::cout << Face(b, c, d).area() << std::endl;
	std::cout << Face(b, c, e).area() << std::endl;
	std::cout << Face(b, d, e).area() << std::endl;
	std::cout << Face(c, d, e).area() << std::endl;


	std::cout << Tetrahedron(a, b, c, d).volume() << std::endl;
	std::cout << Tetrahedron(a, b, c, e).volume() << std::endl;
	std::cout << Tetrahedron(a, b, e, d).volume() << std::endl;
	std::cout << Tetrahedron(a, e, c, d).volume() << std::endl;
	std::cout << Tetrahedron(e, b, c, d).volume() << std::endl;

	std::cout << INCIRCLE(a, b, c, d, e) << std::endl;
};


//Tesselation Incremental(const std::vector<Point>& Vertices = {}) {
//	Tesselation Tess;
//
//	Tess.createBoundingTetrahedron(); // TODO: Add this to addPoints as
//	Tess.addPoints(Vertices);
//	// Tess.deleteBoundingTetrahedron();
//
//
//
//	return Tess;
//};


int main()
{
	
	VERBOSE = false;
	VERBOSE2 = false;  // INCIRCLE and CCW tests
	VERBOSE3 = false;  // Add, remove, change face
	VERBOSE4 = false;  // queue?
	VERBOSE5 = true;   // Point deletion information
	POINT_INDEX_ONLY = true;

	
	std::vector<Point> Vertices;
	int npts = 1000;
	for (int i = 0; i < npts; i++) {
		Vertices.push_back(Point::randomizePoint(i)); 
	};

	
	
	Tesselation Tess;
	clock_t begin = clock();
	
	int i = 1;
	Tess.createBoundingTetrahedron();
	for (const Point& p : Vertices) {
		std::cout << "Adding " << i << "-th point." << std::endl;
		Tess.addPoint(p);
		i++;
	};
	//Tess.deleteBoundingTetrahedron();
	
	clock_t end = clock();
	

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << elapsed_secs << std::endl;
	
	return 0;
}
