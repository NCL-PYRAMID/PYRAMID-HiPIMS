###############################################################################
# HiPIMS sample driver application
# Xue Tong, Robin Wardle
# February 2022
###############################################################################

###############################################################################
# Load Python packages
###############################################################################
import os, sys
import torch
import numpy as np
import datetime
from pythonHipims import CatchFlood_main as catchFlood

###############################################################################
# Add to module path
###############################################################################
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

###############################################################################
# Main function
###############################################################################
def main():
    # Paths setup
    # Base data path set depending on whether in Docker container or not
    platform = os.getenv("HIPIMS_PLATFORM")
    if platform=="docker":
        CASE_PATH = os.getenv("CASE_PATH", "/data")
    else:
        CASE_PATH = os.getenv("CASE_PATH", "./data")
    print(f"Data path: {CASE_PATH}")

    # Input and Output data paths
    RASTER_PATH = os.path.join(CASE_PATH, 'inputs')
    OUTPUT_PATH = os.path.join(CASE_PATH, 'outputs')

    external_rainfall_filename = os.path.join(RASTER_PATH, "HIPIMS", "rain_source.txt")
    if os.path.exists(external_rainfall_filename):
        Rainfall_data_Path = external_rainfall_filename
    else:
        Rainfall_data_Path = os.path.join(RASTER_PATH, 'rain_source_2523.txt')
    print("Using rainfall data: {}".format(Rainfall_data_Path))
    
    Manning = np.array([0.02,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.03])
    hydraulic_conductivity = 0.0
    capillary_head = 0.0
    water_content_diff = 0.0

    Degree = False
    gauges_position = np.array([])
    
    # boundary condition: given Q 
    # this part can be set as parameters on DAFNI
    Qxrange = np.linspace(1.0, 5.0, num = 1000)
    Qyrange = np.linspace(0.0, 0.0, num = 1000)
    Trange = np.linspace(3600, 6*3600, num = 1000)
    givenQ = np.array([Trange, Qxrange, Qyrange]).T
    
    given_Q1 = np.array([[0.0,1.0,0.0]]) 
    given_Q2 = np.array([[12*3600,5.0,0.0]]) 
    given_Q = np.vstack((given_Q1, givenQ, given_Q2))
     
    boundList = {
        'Q_GIVEN': given_Q
    }
    
    boundBox = np.array([[421612.74,563226.97,421620.6,563340.9]])
    bc_type = ['Q_GIVEN']
    default_BC = 'OPEN'
    # end of boundary condition
    
    rasterPath = {
        'DEM_path': os.path.join(RASTER_PATH, 'DEM.tif'),
        'Landuse_path': os.path.join(RASTER_PATH, 'Landuse.tif'),
        'Rainfall_path': os.path.join(RASTER_PATH, 'RainMask.tif')
    }
    landLevel = 0
    
    paraDict = {
       'deviceID': 0,
        'dx': 2.,
        'CFL': 0.5,
        'Manning': Manning,
        'Export_timeStep': 1. * 3600.,        
        't': 0.0,
        'export_n': 0,
        'secondOrder': False,
        'firstTimeStep': 1.0,
        'tensorType': torch.float64,
        'EndTime': 12. * 3600.,        
        'Degree': Degree,
        'OUTPUT_PATH': OUTPUT_PATH,
        'rasterPath': rasterPath,
        'gauges_position': gauges_position,
        'boundBox': boundBox,
        'bc_type': bc_type,
        'landLevel': landLevel,
        'Rainfall_data_Path': Rainfall_data_Path,
        'hydraulic_conductivity': hydraulic_conductivity,
        'capillary_head': capillary_head,
        'water_content_diff': water_content_diff, 
        'default_BC':default_BC,
        'boundBox': boundBox,
        'bc_type': bc_type,
        'boundList':boundList
    }

    catchFlood.run(paraDict)

    # Fix for missing DAFNI metadata file
    # TODO: add this properly!
    title = os.getenv('TITLE', 'PYRAMID <dataset> HiPIMS Simulation Output')
    description = 'Output from HiPIMS simulator'
    geojson = {}
    metadata = f"""{{
      "@context": ["metadata-v1"],
      "@type": "dcat:Dataset",
      "dct:language": "en",
      "dct:title": "{title}",
      "dct:description": "{description}",
      "dcat:keyword": [
        "shetran"
      ],
      "dct:subject": "Environment",
      "dct:license": {{
        "@type": "LicenseDocument",
        "@id": "https://creativecommons.org/licences/by/4.0/",
        "rdfs:label": null
      }},
      "dct:creator": [{{"@type": "foaf:Organization"}}],
      "dcat:contactPoint": {{
        "@type": "vcard:Organization",
        "vcard:fn": "DAFNI",
        "vcard:hasEmail": "support@dafni.ac.uk"
      }},
      "dct:created": "{datetime.datetime.now().isoformat()}Z",
      "dct:PeriodOfTime": {{
        "type": "dct:PeriodOfTime",
        "time:hasBeginning": null,
        "time:hasEnd": null
      }},
      "dafni_version_note": "created",
      "dct:spatial": {{
        "@type": "dct:Location",
        "rdfs:label": null
      }},
      "geojson": {geojson}
    }}
    """
    with open(os.path.join(OUTPUT_PATH, 'metadata.json'), 'w') as f:
        f.write(metadata)

if __name__ == "__main__":
    main()

