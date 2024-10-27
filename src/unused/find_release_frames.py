""" script to find GPS waypoints that are within 100m of release point
- iterate over full list of GPX waypoints
- compare lat/lon of GPS point to lat/lon of release point, using geopy library
- generate list of all GPS waypoints that meet the 100m condition (if no points found, increase to 150m)
- separate the GPS waypoints into acquisition sequences
- find the first and last timestamped GPS waypoints for each acquisition sequence
- using first and last timestamps for each acq. seq. - get the relevant data files from GCP
  # sort the filtrograph_list by acquisition_sequence_id
filtrograph_list.sort(key=lambda x: x['acquisition_sequence_id'])
#acquisition_sequence_id_1 is before release
#acquisition_sequence_id_2 is after release

frame shift  = 1 px 

"""

import os
import csv
import logging
import datetime
from geopy.distance import geodesic
from dateutil import parser
from google.cloud import secretmanager, storage # type: ignore
import psycopg2
import gpxpy
import multisensor_search
import pickle
import numpy as np
import video_creator
import shutil
from typing import Dict, Union, Tuple,List
from skimage import exposure
import cv2
from skimage import filters
import georeferencing

# Logging configuration
logging.basicConfig()
LOG = logging.getLogger("DataDownload")
LOG.setLevel(logging.INFO)


class DataDownload:
    def __init__(
        self, project_id, secret_id, db_name, db_user, db_host, gps_data_dir,main_dir,downloaded_dir, gcp_bucket,release_point
    ):
        self.project_id = project_id
        self.secret_id = secret_id
        self.db_name = db_name
        self.db_user = db_user
        self.db_host = db_host
        self.gps_data_dir = gps_data_dir
        self.main_dir = main_dir
        self.downloaded_dir = downloaded_dir
        self.gcp_bucket = gcp_bucket
        self.conn = self.connect_db()
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # type: ignore
        self.storage_client = storage.Client()
        self.ACQUISITION_SEQUENCES = self.load_acquisition_sequences()
        self.release_point = release_point
        # self.avantes_data_dir = None
        # self.filtrograph_data_dir = None

    def access_secret_version(self, version_id="latest"):
        client = secretmanager.SecretManagerServiceClient()
        secret_name = (
            f"projects/{self.project_id}/secrets/{self.secret_id}/versions/{version_id}"
        )
        response = client.access_secret_version(request={"name": secret_name})
        return response.payload.data.decode("UTF-8")

    def connect_db(self):
        db_pass = self.access_secret_version()
        connection_info = {
            "dbname": self.db_name,
            "user": self.db_user,
            "password": db_pass,
            "host": self.db_host,
        }
        self.connection = psycopg2.connect(**connection_info)
        return self.connection

    def close_connection(self):
        if self.connection:
            self.connection.close()

    def get_release_idx(self):
        query = """
                SELECT a.id as ac_seq_id, a.seq_index,r.release_start_utc, r.release_end_utc, a.timestamp_begin_utc, a.timestamp_end_utc
                FROM release r
                JOIN acquisition_sequence a
                ON a.timestamp_begin_utc between r.release_start_utc
                AND r.release_end_utc
                ORDER BY a.timestamp_begin_utc asc;

                """
        self.cursor.execute(query)
        seq_id_idx = self.cursor.fetchall()
        return seq_id_idx
    
    def get_seq_idx(self):
        query = """
                select id,seq_index,timestamp_begin_utc,timestamp_end_utc
                from acquisition_sequence
                ORDER BY timestamp_begin_utc;
                """
        self.cursor.execute(query)
        seq_id_idx = self.cursor.fetchall()
        return seq_id_idx
    
    

    def load_acquisition_sequences(self):
        sequences = {}
        LOG.info(f"Loading acquisition sequences from CSV file.")
        LOG.info(f"Local dir: {self.gps_data_dir}")

        csv_file_path = os.path.join(self.gps_data_dir ,"acquisition_sequence.csv")

        with open(csv_file_path, "r") as f:
            rdr = csv.DictReader(f)
            sequences = {row["id"]: row for row in rdr}
            for key, seq in sequences.items():
                seq["timestamp_end_utc"] = parser.parse(seq["timestamp_end_utc"])
                seq["timestamp_begin_utc"] = parser.parse(seq["timestamp_begin_utc"])
        return sequences



    def find_nearby_waypoints(self, instrument_name="gpx", max_distance=200) -> List: # type: ignore
            """
            Finds nearby waypoints based on the release point and specified parameters.

            Args:
                release_point (tuple): The coordinates of the release point.
                instrument_name (str, optional): The name of the instrument. Defaults to "gpx".
                max_distance (int, optional): The maximum distance in meters. Defaults to 200.

            Returns:
                tuple: A tuple containing two lists. The first list contains the nearby waypoints, and the second list contains all waypoints.

                
            """
            try:
                with self.connect_db() as conn:
                    with conn.cursor() as cursor:
                        table_map = {"gpx": "airphotonav_gpx"}
                        table_name = table_map.get(instrument_name.lower())

                        if not table_name:
                            raise ValueError(f"Invalid instrument name: {instrument_name}")

                        query = f"""
                        SELECT id, timestamp_utc, lat, lon, acquisition_sequence_id, gcp_bucket_link
                        FROM {table_name};
                        """

                        cursor.execute(query)
                        waypoints = cursor.fetchall()

                nearby_waypoints = [
                    waypoint
                    for waypoint in waypoints
                    if geodesic(self.release_point, (waypoint[2], waypoint[3])).meters
                    <= max_distance
                ]

                nearby_waypoints.sort(key=lambda x: x[4])

                LOG.info(
                    f"Found {len(nearby_waypoints)} waypoints within {max_distance} meters of the release point."
                )
                wp_22 = []
                wp_23 = []
                for wp in nearby_waypoints:
                    timestamp = wp[1].date()

                    if timestamp == datetime.date(2024, 7, 22):
                        wp_22.append(wp)

                    elif timestamp == datetime.date(2024, 7, 23):
                        wp_23.append(wp)
                LOG.info(f"total points found in date {timestamp}: {len(wp_22)}")
                LOG.info(f"total points found in date {timestamp}: {len(wp_23)}")
                LOG.info(f"Nearby waypoints finding process completed.")
                LOG.info("-" * 50)

                return nearby_waypoints

            except Exception as e:
                LOG.error(f"Error fetching data: {e}")
                raise e

  
    def process_determined_waypoints(self, waypoints: List, find_all: bool) -> Dict:
        """Find dataset between first and last timestamped GPS waypoints for each acquisition sequence."""

        if not waypoints:
            LOG.error("No waypoints found")
            raise ValueError("No waypoints found")

        dataset = {}
        trkpt_time_info_location_dict = {}

        # Process each waypoint and extract relevant information
        for waypoint in waypoints:
            acquisition_sequence_id = waypoint[4]
            gcp_bucket_link = waypoint[5]

            trkpt_time_info_location_dict[acquisition_sequence_id] = {}

            abs_gcp_bucket_link = self._get_absolute_gcp_path(gcp_bucket_link)

            try:
                gpx_data = self._parse_gpx_file(abs_gcp_bucket_link)
                time_min, time_max = self._get_time_bounds(gpx_data)

                # Assign min and max times for the acquisition sequence
                trkpt_time_info_location_dict[acquisition_sequence_id] = {
                    "time_min": time_min,
                    "time_max": time_max,
                }

            except FileNotFoundError:
                LOG.error(f"GPX file not found: {abs_gcp_bucket_link}")
                continue
            except Exception as e:
                LOG.error(f"Error processing GPX file: {e}")
                continue

        # Fetch measurements in the time window
        try:
            search = multisensor_search.MultisensorSearch()
            dataset = search.get_measurements_in_time_window(trkpt_time_info_location_dict, find_all)
        except Exception as e:
            LOG.error(f"Error fetching data: {e}")
            if search.conn:
                search.conn.rollback()  # Rollback if there's an error
            raise e

        LOG.info("Fetching dataset process completed.")
        return dataset

    def _get_absolute_gcp_path(self, gcp_bucket_link: str) -> str:
        """Generate the absolute path for the GPX file."""
        abs_gcp_bucket_link = os.path.join(self.gps_data_dir, gcp_bucket_link.split("/")[-1])
        return abs_gcp_bucket_link.replace(os.sep, "/")

    def _parse_gpx_file(self, gcp_bucket_link: str):
        """Open and parse the GPX file."""
        with open(gcp_bucket_link, "r", encoding="utf-8") as gpx_file:
            return gpxpy.parse(gpx_file)

    def _get_time_bounds(self, gpx_data) -> Tuple[datetime.datetime, datetime.datetime]:
        """Extract the minimum and maximum time from GPX tracks."""
        time_min, time_max = None, None
        for trk in gpx_data.tracks:
            for trkseg in trk.segments:
                times = [trkpt.time for trkpt in trkseg.points]  # type: ignore
                time_min = min(times) if not time_min else min(time_min, min(times))
                time_max = max(times) if not time_max else max(time_max, max(times))
        return time_min, time_max # type: ignore




    @staticmethod
    def sanitize_filename(filename):
        LOG.info(filename)
        """Sanitize filename to be valid in Windows file system."""
        return filename.replace(":", "-").replace("/", "-")

    def deneme(self):
        LOG.info("deneme")

    def download_file_from_gcs(self, dataset):
        """
        Download files from Google Cloud Storage, only downloading missing files.

        Args:
            dataset (list): A list of dataset items.

        Returns:
            list: A list of downloaded file paths.
        """
        LOG.info(f"Downloading files from Google Cloud Storage.")
        
        if not dataset:
            LOG.warning("Dataset is empty. No files to download.")
            raise ValueError("Dataset is empty. No files to download.")
            exit()
        
        # LOG.info(self.dir_cdk_prefix)
        # LOG.info("-" * 50)

        data_dirs = {
            '22.07.2024': {
                'avantes': f"{self.downloaded_dir}"+ "/avantes_data_downloaded/avantes_data_22",
                'filtrograph': f"{self.downloaded_dir}"+ "/filtrograph_data_downloaded/filtrograph_data_22"
            },
            '23.07.2024': {
                'avantes': f"{self.downloaded_dir}"+ "/avantes_data_downloaded/avantes_data_23",
                'filtrograph': f"{self.downloaded_dir}"+ "/filtrograph_data_downloaded/filtrograph_data_23"
            }
        }
        # Create directories if they don't exist
        for date in data_dirs:
            for sensor_type in data_dirs[date]:
                LOG.info(data_dirs[date][sensor_type])
                os.makedirs(data_dirs[date][sensor_type], exist_ok=True)



        # Function to download files
        def download_files(data_list, local_dir):
            for data in data_list:
                gcp_bucket_link = data[2]
                sanitized_filename = self.sanitize_filename(os.path.basename(gcp_bucket_link))
                local_file_path = os.path.join(local_dir, sanitized_filename)
                blob = self.storage_client.bucket(self.gcp_bucket).blob(gcp_bucket_link)
                LOG.info(f"Downloading {gcp_bucket_link} to {local_file_path}")
                blob.download_to_filename(local_file_path)


        # Process the dataset
        for seq_id in dataset:
            for date, sensors in dataset[seq_id].items():
                # LOG.info(data_dirs[date]['filtrograph'])
                # LOG.info(data_dirs[date]['avantes'])
                download_files(sensors['filtrograph'], data_dirs[date]['filtrograph'])
                download_files(sensors['avantes'], data_dirs[date]['avantes'])


    

    def get_release_times(self):
        """
        release_data_dict = {
            "id": {
                "date": "2024-07-22",
                "release_start_stabilised_utc": "2024-07-22 12:00:00",
                "release_end_utc": "2024-07-22 12:30:00"
            }
        }
        
        """

        release_data_dict = {}
        try:
            query_relsease = """
            SELECT id,date,release_start_stabilised_utc,release_end_utc

            FROM release
            ;
            """
            # Execute the queries for each time window
            self.cursor.execute(query_relsease)
            release_data = self.cursor.fetchall()
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
        
        for id in release_data:
            release_data_dict[id[0]] = {
                "date": id[1],
                "release_start_stabilised_utc": id[2],
                "release_end_utc": id[3]
            }

        return release_data_dict
    
 

LOG = logging.getLogger("dataProcessing")
LOG.setLevel(logging.INFO)
class dataProcessing:
    def __init__(self, dataset, nearby_waypoints,downloaded_dir):
        # self.downloaded_files = downloaded_files
        self.dataset = dataset
        self.nearby_waypoints = nearby_waypoints
        self.downloaded_dir = downloaded_dir
        self.release_data_dict = downloader.get_release_times()
        self.avantes_data_dir = os.path.join(self.downloaded_dir, "avantes_data_downloaded")
        self.filtrograph_data_dir = os.path.join(self.downloaded_dir, "filtrograph_data_downloaded")
        self.width = 640
        self.length = 512
        self.release_idx = downloader.get_release_idx()
        self.seq_idx = downloader.get_seq_idx()
        self.worlview3_dir_prefix = "D:\07_AIRMO\02_work\01_airborne_data_processing_repo\airborne_data_processing\data"
        self.worlview3_dir_prefix = self.worlview3_dir_prefix.replace(os.sep, "/")
        
        

    def filtrograph_generator(self):
        # Klasör yollarını oluşturma
        filtrograph_dir_prefixes = {
            "22.07.2024": os.path.join(self.filtrograph_data_dir, "filtrograph_data_22").replace(os.sep, "/"),
            "23.07.2024": os.path.join(self.filtrograph_data_dir, "filtrograph_data_23").replace(os.sep, "/")
        }

        LOG.info(filtrograph_dir_prefixes["22.07.2024"])
        LOG.info(filtrograph_dir_prefixes["23.07.2024"])

        # Kullanıcıdan görsel denetim isteği al
        # ask = input("Do you want to do visual inspection? (y/n): ").strip().lower()
        ask = "n"
        if ask != "y":
            LOG.info("Passing the visual inspection")

        # Dataset içindeki her acquisition ID ve tarih için işlemleri yap
        for acq_id, dates in self.dataset.items():
            for date, sensor_data in dates.items():
                if date in filtrograph_dir_prefixes:
                    filtrograph_dir_prefix = filtrograph_dir_prefixes[date]
                    self._process_filtrograph_files(acq_id, filtrograph_dir_prefix, sensor_data['filtrograph'], ask)

   


    def _process_filtrograph_files(self, acq_id, filtrograph_dir_prefix, filtrograph_data, ask):
        # Klasördeki tüm alt dizinleri ve dosyaları yürü
        for root, dirs, files in os.walk(filtrograph_dir_prefix):
            for dir in dirs:
                if acq_id in dir:
                    dir_path = os.path.join(filtrograph_dir_prefix, dir)
                    self._process_files_in_dir(dir_path, filtrograph_data, ask)


    def _process_files_in_dir(self, dir_path, filtrograph_data, ask):
        # İlgili klasördeki tüm .npy dosyalarını işle
        output_folder = os.path.join(dir_path, "tif")
        os.makedirs(output_folder, exist_ok=True)  # Klasörü bir kez oluştur

        for file_name in os.listdir(dir_path):
            if file_name.endswith(".npy"):
                file_path = os.path.join(dir_path, file_name).replace(os.sep, "/")
                for data in filtrograph_data:
                    if file_name == data[2].split("/")[-1]:
                        img = np.load(file_path).reshape(self.length, self.width)
                        yield img, file_name, file_path 

                        if ask == "y":
                            self._perform_visual_inspection(file_path, output_folder, file_name)



    def _perform_visual_inspection(self, img, output_folder, file_name):
        # Görsel denetim işlemini gerçekleştir
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98)) # type: ignore
        img = (((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)).astype(np.uint8)

        # Görüntüyü TIFF formatında kaydet
        output_path = os.path.join(output_folder, file_name.replace(".npy", ".tif")).replace(os.sep, "/")

        if not os.path.exists(output_path):
            yield cv2.imwrite(output_path, img)
        else:
            LOG.info(f"{file_name} already exists")

    """
    # def read_filtrograph_dataset(self, dataset):

    #     filtrograph_22_dir_prefix = os.path.join(self.filtrograph_data_dir, "filtrograph_data_22").replace(os.sep, "/")
    #     filtrograph_23_dir_prefix = os.path.join(self.filtrograph_data_dir, "filtrograph_data_23").replace(os.sep, "/")
    #     LOG.info(filtrograph_22_dir_prefix)
    #     LOG.info(filtrograph_23_dir_prefix)
    #     ask = input("Do you want to do visual inspection? (y/n): ")
    #     if ask is not "y":
    #         LOG.info("Passing the visual inspection")


    #     for acq_id in dataset:
    #         for date in dataset[acq_id]:
    #             if date == "22.07.2024":
    #                 for root, dirs, files in os.walk(filtrograph_22_dir_prefix):
    #                     for dir in dirs:
    #                         if acq_id in dir:
    #                             # LOG.info(f"Found {acq_id} in {dir}")
    #                             for file_name in os.listdir(os.path.join(filtrograph_22_dir_prefix, dir)):
    #                                 if file_name.endswith(".npy"):
    #                                     file_path = os.path.join(filtrograph_22_dir_prefix, dir, file_name).replace(os.sep, "/")
    #                                     for data in dataset[acq_id][date]['filtrograph']:
    #                                         if file_name == data[2].split("/")[-1]:
    #                                             LOG.info(file_name)
    #                                             LOG.info(data[2])
    #                                             if ask == "y":
    #                                                 for data in dataset[acq_id][date]['filtrograph']:
    #                                                     data[2]= img = np.load(file_path).reshape(self.length, self.width)
                                                    
    #                                                     p2,p98 = np.percentile(img, (2,98))
    #                                                     img = exposure.rescale_intensity(img, in_range=(p2,p98))
    #                                                     img = (((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)).astype(np.uint8)
    #                                                     output_folder = os.makedirs(os.path.join(filtrograph_22_dir_prefix, dir, "tif"), exist_ok=True)
    #                                                     output_path = os.path.join(filtrograph_22_dir_prefix, dir, "tif", file_name.replace(".npy", ".tif")).replace(os.sep, "/")
    #                                                     if not os.path.exists(output_path):
    #                                                         cv2.imwrite(output_path, img)
    #                                                     else:
    #                                                         LOG.info(f"{file_name} already exists")

                                    
    #             elif date == "23.07.2024":
    #                 for root, dirs, files in os.walk(filtrograph_23_dir_prefix):
    #                     for dir in dirs:
    #                         if acq_id in dir:
    #                             # LOG.info(f"Found {acq_id} in {dir}")
    #                             for file_name in os.listdir(os.path.join(filtrograph_23_dir_prefix, dir)):
    #                                 if file_name.endswith(".npy"):
    #                                     file_path = os.path.join(filtrograph_23_dir_prefix, dir, file_name).replace(os.sep, "/")
    #                                     for data in dataset[acq_id][date]['filtrograph']:
    #                                         if file_name == data[2].split("/")[-1]:
    #                                             LOG.info(file_name)
    #                                             LOG.info(data[2])
    #                                             if ask == "y":
    #                                                 for data in dataset[acq_id][date]['filtrograph']:
    #                                                     data[2]= img = np.load(file_path).reshape(self.length, self.width)
                                                    
    #                                                     p2,p98 = np.percentile(img, (2,98))
    #                                                     img = exposure.rescale_intensity(img, in_range=(p2,p98))
    #                                                     img = (((img - np.min(img)) / (np.max(img) - np.min(img)) * 255)).astype(np.uint8)
    #                                                     output_folder = os.makedirs(os.path.join(filtrograph_22_dir_prefix, dir, "tif"), exist_ok=True)
    #                                                     output_path = os.path.join(filtrograph_22_dir_prefix, dir, "tif", file_name.replace(".npy", ".tif")).replace(os.sep, "/")
    #                                                     if not os.path.exists(output_path):
    #                                                         cv2.imwrite(output_path, img)
    #                                                     else:
    #                                                         LOG.info(f"{file_name} already exists")
                                                        
    #     return dataset
        
"""

    


# Example usage to run the script
if __name__ == "__main__":
    project_id = "1045419825555"
    secret_id = "airborne2024_db_pass"
    db_name = "postgres"
    db_user = "postgres"
    db_host = "34.32.68.133"
    gps_data_dir = "D:/07_AIRMO/02_work/01_airborne_data_processing_repo/airborne_data_processing/data/airphotonav/all"
    data_repo_dir = "D:/07_AIRMO/02_work/01_airborne_data_processing_repo/airborne_data_processing/data"
    downloaded_dir = "D:/07_AIRMO/02_work"
    gcp_bucket = "tadi_july"
    release_point = (43.4128056, -0.642667)
    reference_image_path = "D:/07_AIRMO/02_work/worldview_3_test_site.tif"
    output_dir = "D:/07_AIRMO/02_work/03_georef_output"
    os.makedirs(output_dir, exist_ok=True)

    downloader = DataDownload(
        project_id, secret_id, db_name, db_user, db_host, gps_data_dir,data_repo_dir,downloaded_dir, gcp_bucket,release_point
    )

    
    try:
        with open(os.path.join(data_repo_dir,"nearby_waypoints.pckl"), 'rb') as f:
            LOG.info("nearby waypoints found")
            nearby_waypoints = pickle.load(f) 
            LOG.info("nearby waypoints loaded")
    except Exception as err:

        nearby_waypoints = downloader.find_nearby_waypoints()
        LOG.error(f"could not find nearby waypoints locally, recalculated!")
        with open(os.path.join(data_repo_dir,"nearby_waypoints.pckl"), 'wb') as f:
            pickle.dump(nearby_waypoints, f)
    
    # ask = input("Do you want to search all data without timeframe? (y/n): ")
    ask = "y"
    if ask == "y":
        ask = True
        try:
            with open(os.path.join(data_repo_dir,"dataset_all.pckl"), 'rb') as f:
                LOG.info("all dataset found")
                dataset = pickle.load(f) 
                LOG.info("all dataset loaded")
        except Exception as err:
            dataset = downloader.process_determined_waypoints(nearby_waypoints,find_all=ask)
            LOG.error(f"could not find dataset locally, recalculated!")
            with open(os.path.join(data_repo_dir,"dataset_all.pckl"), 'wb') as f:
                pickle.dump(dataset, f)        
    else:
        ask = False
        try:
            with open(os.path.join(data_repo_dir,"dataset_nearby.pckl"), 'rb') as f:
                LOG.info("nearby dataset found")
                dataset = pickle.load(f) 
                LOG.info("nearby dataset loaded") 
        except Exception as err:
            dataset = downloader.process_determined_waypoints(nearby_waypoints,find_all=ask)
            LOG.error(f"could not find dataset locally, recalculated!")
            with open(os.path.join(data_repo_dir,"dataset_nearby.pckl"), 'wb') as f:
                pickle.dump(dataset, f)
    # ask = input("Do you want to download the files? (y/n): ")
    ask = "n"
    if ask == "y":
        again_ask = input("Are you sure? (y/n): ")
        if again_ask == "y":
            again_ask_v2 = input("This is the last destination? (y/n): ")
            if again_ask_v2 == "y":
                downloader.download_file_from_gcs(dataset)
    else:
        LOG.info("Files are not downloaded")
    
    # Data Processing Section
    # ask = input("Would you like to inspect images visually? (y/n): ")
    # if ask == "y":
    process = dataProcessing(dataset=dataset, nearby_waypoints=nearby_waypoints,downloaded_dir=downloaded_dir)
    georef = georeferencing.Georeferencing(dataset,reference_image_path, output_dir,downloaded_dir)
    georef.filtrograph_generator()
    downloader.deneme()


