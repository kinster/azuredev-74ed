import base64

with open("./inputs/images/CommercialBuildingInternalWallTypesGroundFloorLevel01FloorPlans-1348828089063799274_markedup_page1.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    print(encoded_string)

