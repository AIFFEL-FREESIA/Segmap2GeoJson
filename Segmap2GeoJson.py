import Inference
import click
import cv2
import os
import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon
from shapely import affinity

def denoise(img):
    def _denoise(mask, eps):
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)

    def _grow(mask, eps):
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct)

    def _erode(mask, esp):
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (esp, esp))
        return cv2.erode(mask, struct)

    _img = _erode(img, 5)
    _img = _denoise(img, 7)
    _img = _grow(_img, 3)
    _img = _denoise(_img, 2)
    return _img

def contour_to_polygon(contours):
    pts = np.array([[point[0] for point in contour] for contour in contours ])
    polygons = [Polygon(p) for p in pts]
    return polygons

def pix_coor_to_real_coor(site_geo, src, pix_size):
    real_URC = (site_geo.bounds[2], site_geo.bounds[3])
    real_LLC = (site_geo.bounds[0], site_geo.bounds[1])

    pix_width = pix_size # 1024*1024 고정
    pix_height = pix_size

    width_scale = abs((real_URC[0] - real_LLC[0]) / pix_width)
    height_scale = abs((real_URC[1] - real_LLC[1]) / pix_height)

    width_trans = real_LLC[0]
    height_trans = real_LLC[1]
        
    mirrored = affinity.affine_transform(src, [1, 0, 0, -1, 0, pix_height ])
    return affinity.affine_transform(mirrored, [width_scale, 0, 0, height_scale, width_trans, height_trans ])

@click.command()
@click.option('--img_path', help='Satellite image path')
@click.option('--bounds', help='Coordinates of satellite image bounds, format should be "x,y x,y x,y x,y')
@click.option('--save_path', help='GeoJson file path')
@click.option('--crs', default="EPSG:4326")
def img_to_geojson(img_path, bounds, save_path, crs):
    try:
        if not os.path.isfile(img_path):
            raise Exception(f'[Error] No such file or directory: {img_path}')
        if os.path.splitext(img_path)[1] not in ['.png', '.jpg']:
            raise Exception(f'[Error] Input file should be .png or .jpg: {img_path}')
    except Exception as e:
        print(e)
        return

    # inference
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    seg_map = Inference.inference(img)
    
    # split seg map
    COLOR_MAP = [
        (165, 42, 42),
        (0, 192, 0),
        (255,255,255)
    ]
    seg_maps = [(seg_map == v) for v in COLOR_MAP]
    seg_maps = [(mask[:,:,0] * mask[:,:,1] * mask[:,:,2]) for mask in seg_maps]

    # denoise
    seg_maps = [denoise(seg_map.astype(np.uint8)) for seg_map in seg_maps]

    # segmentation map → contours
    building_contours, building_hierachy  = cv2.findContours(seg_maps[0].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    road_contours, road_hierachy  = cv2.findContours(seg_maps[1].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # contours → shapely.geometry.Polygon → affine transform(pixel to real)
    # get site polygon
    bounds_coordinates = [coors.split(',')[:2] for coors in bounds.split(' ')]
    bounds_coordinates = [[float(coors[0]), float(coors[1])] for coors in bounds_coordinates]
    bounds_coordinates.append(bounds_coordinates[0])
    site_polygon = Polygon(bounds_coordinates)

    building_polygons = contour_to_polygon(building_contours)
    road_polygons = contour_to_polygon(road_contours)

    building_real_coor_polygons = [pix_coor_to_real_coor(site_polygon, polygon, 1024) for polygon in building_polygons]
    road_real_coor_polygons = [pix_coor_to_real_coor(site_polygon, polygon, 1024) for polygon in road_polygons]

    # add hole
    building_ptr = [ h[-1] for h in building_hierachy[0]]
    road_ptr = [ h[-1] for h in road_hierachy[0]]

    for i, ptr in enumerate(building_ptr):
        if ptr > -1:
            building_real_coor_polygons[ptr] = building_real_coor_polygons[ptr].difference(building_real_coor_polygons[i])
    
    for i, ptr in enumerate(road_ptr):
        if ptr > -1:
            road_real_coor_polygons[ptr] = road_real_coor_polygons[ptr].difference(road_real_coor_polygons[i])
    

    # 5. GeoDataFrame 생성
    data = {'type': ['building']*len(building_real_coor_polygons) + 
                    ['road']*len(road_real_coor_polygons), 
            'geometry': building_real_coor_polygons + road_real_coor_polygons,
            'is_interior': building_ptr + road_ptr}
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    # hole 제거
    interior = gdf[gdf['is_interior'] > -1].index
    gdf.drop(interior , inplace=True)
    gdf.drop('is_interior' , axis=1, inplace=True)

    # 6. projection 후 면적 계산 : E22 참고
    center_point_x, center_point_y = site_polygon.centroid.coords.xy
    gdf_proj = gdf.to_crs(f"+proj=cea +lat_0={center_point_y[0]} +lon_0={center_point_x[0]} +units=m")
    gdf_proj['area'] = gdf_proj['geometry'].area

    # 7. EPSG:4326으로 다시 좌표계 변환
    gdf_proj_4326 = gdf_proj.to_crs(epsg="4326")

    # 8. geojson 저장
    gdf_proj_4326.to_file(save_path, driver="GeoJSON") 

if __name__ == '__main__':
    img_to_geojson()