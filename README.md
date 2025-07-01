# smart-traffic-monitoring-and-aqi-estimation

This project is a Python + OpenCV based real-time traffic monitoring system that can detect and classify vehicles, track their movement direction, calculate speed, detect overspeeding, estimate air quality index (AQI), and visualize all the data using live charts.

**Features:**

    1.Detects and classifies vehicles as Cars or Trucks using background subtraction and bounding box analysis.
    2.Tracks vehicle movement direction (UP or DOWN) across predefined lines.
    3.Calculates vehicle speed using frame rate and pixel-to-meter conversion.
    4.Flags overspeeding vehicles (above 100 km/h) and saves their images.
     5.Estimates AQI based on vehicle counts and emission rates.
    6.Displays live bar chart of vehicle stats and AQI line graph using matplotlib.
    7.Saves all results including graphs and AQI report at the end.


**Technologies Used:**

    Python 3
     OpenCV
     NumPy
     Matplotlib
     Background Subtraction (MOG2)
     Threading



**How It Works:**


    1.The system reads a video file and applies background subtraction to isolate moving vehicles.
    2.Contours are detected and bounding boxes are drawn around each vehicle.
    3.Vehicle direction (UP or DOWN) is determined by analyzing movement across lines.
    4.Each vehicle is classified as a Car or Truck based on area and aspect ratio.
    5.Speed is calculated using frame counts and pixel distance, converted to km/h.
    6.Overspeeding vehicles are identified and their images are saved.
    7.AQI is calculated based on total emissions from cars and trucks.
    8.Two live plots are updated: a bar chart of vehicle stats and a line graph of AQI over time.
    9.All results are saved when the video end


