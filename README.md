# Segmap2GeoJson
모델에서 검출된 건물과 도로의 segmentation map을 GeoJson 혹은 이외의 포맷을 변환한다.

<br>

### Segmap2GeoJson.py Usage
```bash
python3 Segmap2GeoJson.py --img_path={Satellite image path} 
                          --bounds={Coordinates of satellite image bounds, format should be "x,y x,y x,y x,y"}
                          --save_path={GeoJson file path}
```
[![LV4 Youtube](https://img.youtube.com/vi/dTabdwIrecY/0.jpg)](https://www.youtube.com/watch?v=dTabdwIrecY)
