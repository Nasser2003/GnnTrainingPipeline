from datetime import datetime


class Utils:

    @staticmethod
    def persist_to_file(obj, file_path: str = './graph', format: str = "csv"):
        if format == "csv":
            obj.to_csv(file_path + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".csv", index=False, mode='a')
        elif format == "gml":
            obj.save(file_path + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".gml")
        elif format == "json":
            obj.to_json(file_path + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".json", lines=True, orient='records', mode='a')
