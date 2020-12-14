#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv\cv.hpp"
#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

#define ESCAPE 27

Mat procesamiento(Mat img, Mat imgHSV, int myColors[6]);
void detectarCaras(Mat &img, CascadeClassifier detector, Mat imgGris, Mat eqHist, int &count);
void pintar(Mat& img, Mat &imgProces, Mat &imgAux, Mat &imgAuxMask, int myColorsValues[3], Point& prevPoint, Point &currPoint);

int main(int argc, char* argv[]) {
	//Objetos del programa
	Mat img, imgHSV, imgGris, eqHist, portada;
	Mat imgProcesAzul, imgProcesRojo, imgProcesVerde; //Mascaras para cada color
	Mat imgAuxAzul, imgAuxRojo, imgAuxVerde;
	Mat imgAuxMaskAzul, imgAuxMaskRojo, imgAuxMaskVerde;
	int numSavedImg = 0;

	//Inicializa captura de video
	VideoCapture capture(0);
	char pressedKey = 0;
	bool success;
	//Activacion HaarCascade
	CascadeClassifier detector;

	Point prevPointAzul = Point(-1,-1), prevPointRojo = Point(-1, -1), prevPointVerde = Point(-1, -1);
	Point currPointAzul, currPointRojo, currPointVerde;

	// Valores obtenidos de las barras de ajuste
	int myColors[3][6] = {
		{96,61,170,120,147,255}, //azul
		{150,61,88,179,122,255},  //rosa
		{69,77,97,102,255,255}  //verde
	};

	// Valores del color en BGR con los que se va a pintar
	int myColorsValues[3][3] = {
		{255,178,102}, //azul
		{203,192,255},  //rosa    BGR
		{0,255,128}  //verde
	};

	portada = imread("Portada.jpg", cv::IMREAD_COLOR);

	if (!portada.data) {
		cout << "error loading image" << endl;
		return 1;
	}

	//Comprueba que el video este disponible
	if (!capture.isOpened()) {
		cout << "Error in loading the video!" << endl;
	}
	else {
		const String windows[] = { "Pizarra", "Portada" };
		cvui::init(windows, 2);

		//Comprobacion HaarCascade
		if (!detector.load("haarcascade_frontalface_alt.xml"))
			cout << "No se puede abrir clasificador." << endl;

		while (true) { //EJECUCION DE PORTADA DE INICIO
			cvui::context("Portada");
			if (cvui::button(portada, 285, 290, "Jugar", 0.5, 0x2679ff)) {
				destroyWindow("Portada");
				break;
			}
			cvui::imshow("Portada", portada);
			waitKey(1);
		}
		while (true) { //EJECUCION DE PROGRAMA PRINCIPAL: PIZARRA
			TickMeter tickMeter;
			tickMeter.start();
			success = capture.read(img);
			if (success == false) {
				cout << "Cant read from file!" << endl;
				return 1;
			}

			if (imgAuxAzul.dims==0) imgAuxAzul = Mat(img.size(), CV_8UC3, Scalar(0)); //Si la imagen auxiliar esta vacia, se crea en negro del tamanyo de img
			if (imgAuxRojo.dims == 0) imgAuxRojo = Mat(img.size(), CV_8UC3, Scalar(0)); //Si la imagen auxiliar esta vacia, se crea en negro del tamanyo de img
			if (imgAuxVerde.dims == 0) imgAuxVerde = Mat(img.size(), CV_8UC3, Scalar(0)); //Si la imagen auxiliar esta vacia, se crea en negro del tamanyo de img
			
			// -------------------------- PROCESAMIENTO DE LA IMAGEN ----------------------------//
			flip(img, img, 1); //La imagen se gira para que quede en modo espejo
			cvtColor(img, imgHSV, COLOR_BGR2HSV); //Cambio de RGB a HSV
			imgProcesAzul = procesamiento(img, imgHSV, myColors[0]); //imgProces es la mascara con el rotulador azul en blanco
			imgProcesRojo = procesamiento(img, imgHSV, myColors[1]); //imgProces es la mascara con el rotulador rojo en blanco
			imgProcesVerde = procesamiento(img, imgHSV, myColors[2]); //imgProces es la mascara con el rotulador verde en blanco
			
			// ------------------------- EXTRACCION DE CARACTERISTICAS ----------------------------//
			int count;
			cvtColor(img, imgGris, CV_BGR2GRAY); //Cambio de RGB a escala de grises
			equalizeHist(imgGris, eqHist); //Histograma ecualizado
			detectarCaras(img, detector, imgGris, eqHist, count); //Detecta caras y pinta rectángulo

			pintar(img, imgProcesAzul, imgAuxAzul, imgAuxMaskAzul, myColorsValues[0], prevPointAzul, currPointAzul); //Detecta bordes en mask y pinta sobre img como si fuera una pizarra
			pintar(img, imgProcesRojo, imgAuxRojo, imgAuxMaskRojo, myColorsValues[1], prevPointRojo, currPointRojo);
			pintar(img, imgProcesVerde, imgAuxVerde, imgAuxMaskVerde, myColorsValues[2], prevPointVerde, currPointVerde);

			cvui::context("Pizarra");
			if (cvui::button(img, 240, 450, "Cerrar Pizarra", 0.4, 0x2679ff)) { //BOTON PARA CERRAR PROGRAMA
				break;
			}
			if (cvui::button(img, 517, 1, "Borrar Pizarra", 0.4, 0xffd635)) { //BOTON PARA BORRAR PIZARRA
				imgAuxAzul = Mat(img.size(), CV_8UC3, Scalar(0));
				imgAuxRojo = Mat(img.size(), CV_8UC3, Scalar(0));
				imgAuxVerde = Mat(img.size(), CV_8UC3, Scalar(0));
			}
			if (cvui::button(img, 1, 1, "Guardar Pizarra", 0.4, 0x87ef4f)) { //BOTON PARA HACER CAPTURA DE PANTALLA
				imwrite("Capturas/capturaPizarra" + to_string(numSavedImg) + ".png", img);
				numSavedImg++;
			}
			cvui::imshow("Pizarra", img);

			imshow("imgAUx",imgAuxRojo);

			waitKey(1);
			tickMeter.stop();
			cout << "FRAMES PER SEC:" << 1000/tickMeter.getTimeMilli() << endl;
		}

		destroyAllWindows;
		capture.release();
		return 0;
	}
}

Mat procesamiento(Mat img, Mat imgHSV, int myColors[6]) {
	Mat imgProces;
	inRange(imgHSV, Scalar(myColors[0], myColors[1], myColors[2]), Scalar(myColors[3], myColors[4], myColors[5]), imgProces); //Crea mascara de color
	medianBlur(imgProces, imgProces, 9); //Elimina ruido
	Mat kernel = getStructuringElement(0, Size(3, 3)); //Crea kernel cuadrado 3x3
	dilate(imgProces, imgProces, kernel); //Dilate + erode -> opening de la imagen para eliminar errores pimienta
	erode(imgProces, imgProces, kernel);
	return imgProces;
}

void detectarCaras(Mat& img, CascadeClassifier detector, Mat imgGris, Mat eqHist, int& count) {
	vector<Rect> rect; //Vector de las caras
	detector.detectMultiScale(eqHist, rect, 1.2, 3, 0, Size(200, 200)); //Detector de caras
	count = 0;//Inicio contador player
	for (Rect rc : rect) {
		//rectangle(img, Point(rc.x, rc.y), Point(rc.x + rc.width, rc.y + rc.height), CV_RGB(0, 255, 0), 2); //Pintar rectangulo en la cara al detectar
		count++;//contador =contador +1
		std::string cadena = "";//pasar a char
		cadena = std::to_string(count);
		cv::putText(img, "Player: ", cv::Point(rc.x - 10, rc.y - 10), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 143, 143), 2);
		cv::putText(img, cadena, cv::Point(rc.x + 110, rc.y - 10), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 143, 143), 2); //Numero de jugador en pantalla
	}
}

void pintar(Mat &img, Mat &imgProces, Mat& imgAux, Mat& imgAuxMask, int myColorsValues[3], Point &prevPoint, Point& currPoint) {
	int drawLine;
	vector<vector<Point>> contours; //Guarda contornos de la mascara procesada
	Point penTip; //Guarda coordenadas de la punta del boli
	vector<Vec4i> hierarchy;
	findContours(imgProces, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE); //Extrae los contornos
	vector<vector<Point>>hull(contours.size()); //Guarda los cercos convexos de cada contorno
	for (int idx = 0; idx < contours.size(); idx++) { //Iteracion por cada contorno
		if (contourArea(contours[idx]) > 1000) {
			convexHull(contours[idx], hull[idx]); //Cerco convexo
			int size = hull[idx].size();
			int minYposit = 0;
			for (int i = 0; i < size; i++) {
				if (hull[idx][i].y < hull[idx][minYposit].y) { //Buscamos punto mas pequenyo en el eje de la y
					minYposit = i;
				}
			}
			Point penTip = hull[idx][minYposit];
			currPoint = penTip;

			if (prevPoint.x!=-1){
				line(imgAux, prevPoint, currPoint, Scalar(myColorsValues[0], myColorsValues[1], myColorsValues[2]), 6);
			}
			prevPoint = currPoint;
			drawContours(img, hull, idx, Scalar(myColorsValues[0], myColorsValues[1], myColorsValues[2])); //DIbujamos cerco convexo en el boli
		}
		else {
			prevPoint = Point(-1, -1);
		}
	}

	cvtColor(imgAux, imgAuxMask, COLOR_BGR2GRAY);
	threshold(imgAuxMask, imgAuxMask, 10, 255, THRESH_BINARY);
	imgAux.copyTo(img, imgAuxMask); //Union del trazo del boli con la imagen de la camara
}