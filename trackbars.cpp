#include "opencv\cv.hpp"
#include <iostream>


using namespace cv;
using namespace std;

#define ESCAPE 27
#define pass (void)0 // Para usar pass en C++

// Función vacía auxiliar
static void empty(int, void*)
{
	pass;
}

////////////////////////////
//// PROGRAMA PRINCIPAL ////
////////////////////////////

int main(int argc, char* argv[])
{
	/////////////////////// Declaraciones de variables //////////////////////////

	// Parametros de la ventana
	const int frameWidth = 640; // Ancho de la ventana
	const int frameHeight = 480; // Alto de la ventana
	const int brightness = 20; //Brillo de la ventana

	// Valores iniciales de las barras de ajuste
	int low_H = 0, high_H = 179, low_S = 0, high_S = 255, low_V = 0, high_V = 255;

	// Variables para guardar los valores ajustados
	int h_min, h_max, s_min, s_max, v_min, v_max;

	// Imagenes con las que se va a trabajar
	Mat img, imgGirada, imgHSV, mask, concatV;

	// teclado pulsado
	char pressedKey = 0;

	// variable auxiliar para comprobar el exito de la lectura de video
	bool success;

	////////////////////////////// Programa ////////////////////////////////////

	// Creamos la ventana con las barras de ajuste
	namedWindow("TrackBars");
	resizeWindow("TrackBars", 400, 310);

	createTrackbar("Hue Min", "TrackBars", &low_H, 179, empty);
	createTrackbar("Hue Max", "TrackBars", &high_H, 179, empty);
	createTrackbar("Sat Min", "TrackBars", &low_S, 255, empty);
	createTrackbar("Sat Max", "TrackBars", &high_S, 255, empty);
	createTrackbar("Val Min", "TrackBars", &low_V, 255, empty);
	createTrackbar("Val Max", "TrackBars", &high_V, 255, empty);

	// Tomamos la camara como entrada de video
	VideoCapture capture(0);

	// Ajustamos los parametros de video
	capture.set(3, frameWidth);
	capture.set(4, frameHeight);
	capture.set(10, brightness);


	// Comprobamos si la camara es accesible
	if (!capture.isOpened())
	{
		cout << "Error in loading the cam!" << endl;
		return 1;
	}

	// Bucle que se repite mientras no se pulse la tecla Esc
	while (waitKey(1) && pressedKey != ESCAPE)
	{
		// leemos frame a frame en bucle
		success = capture.read(img);

		// Comprobamos si se lee adecuadamente
		if (success == false)
		{
			cout << "Cant read from cam!" << endl;
			return 1;
		}

		// Asignamos los valores actuales de las barras de ajuste a las variables y los imprimimospor pantalla
		h_min = getTrackbarPos("Hue Min", "TrackBars");
		h_max = getTrackbarPos("Hue Max", "TrackBars");
		s_min = getTrackbarPos("Sat Min", "TrackBars");
		s_max = getTrackbarPos("Sat Max", "TrackBars");
		v_min = getTrackbarPos("Val Min", "TrackBars");
		v_max = getTrackbarPos("Val Max", "TrackBars");
		cout << h_min << " " << h_max << " " << s_min << " " << s_max << " " << v_min << " " << v_max << "\n";

		// Giramos la imagen para evitar el efecto espejo
		flip(img, imgGirada, 1);

		// Obtenemos la imagen en HVS para detectar los colores
		cvtColor(imgGirada, imgHSV, COLOR_BGR2HSV);

		// Creamos una mascara en la que se debe ver, mediante ajuste, todo en negro menos aquello del color a buscar
		inRange(imgHSV, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), mask);

		// Reducimos al 50% el tamaño de la imagen recibida por la camara y de la HSV, y los juntamos en una misma pantalla
		resize(imgGirada, imgGirada, Size(), 0.5, 0.5);
		resize(imgHSV, imgHSV, Size(), 0.5, 0.5);
		vconcat(imgGirada, imgHSV, concatV);

		// Ponemos en cada ventana un texto indicativo
		putText(concatV, "Camara", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
		putText(concatV, "HSV", Point(10, 260), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
		//putText(mask, "Det.Color", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

		// Suavizamos los bordes
		medianBlur(mask, mask, 9); //Elimina ruido
		Mat kernel = getStructuringElement(0, Size(3, 3)); //Crea kernel cuadrado 3x3
		dilate(mask, mask, kernel); //Dilate + erode -> opening de la imagen para eliminar errores pimienta
		erode(mask, mask, kernel);
		//medianBlur(mask, mask, 3);

		Mat img2;
		cvtColor(mask,img2,COLOR_GRAY2BGR);
		vector<vector<Point>> contours; //Guarda contornos de la mascara procesada
		Point penTip; //Guarda coordenadas de la punta del boli
		vector<Vec4i> hierarchy;
		findContours(mask, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE); //Extrae los contornos
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
				circle(img2,penTip,5,Scalar(203, 192, 255),2);

				drawContours(img2, hull, idx, Scalar(203, 192, 255),3); //DIbujamos cerco convexo en el boli
			}
		}

		// Mostramos las imagenes
		imshow("cam", concatV);
		imshow("mask", img2);

		// Actualizamos el valor de la variable pressedKey
		pressedKey = waitKey(1);

	}

	// Liberamos memoria
	destroyAllWindows();
	capture.release();
	return 0;
}