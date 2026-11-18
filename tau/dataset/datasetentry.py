from typing import List

from .datasetinstance import DatasetInstance


class DatasetEntry:
    def __init__(
            self,
            instances: List[DatasetInstance] = None,
            date: int = None,
            Braak1: float = None,
            Braak2: float = None,
            Braak3: float = None,
            Braak4: float = None,
            Braak5: float = None,
            Braak6: float = None,
            Braak1to4: float = None,
            Braak1_2: float = None,
            Braak3_4: float = None,
            Braak5_6: float = None,
            MetaTemporal: float = None,
            tracer: str = None,
            patient_id: str = None,
            dataset: str = None,
            age: float = None,
            sex: int = None,
            apoe: int = None,
            edu: int = None,
            mmse: int = None,
            status: str = None,
            ):
        self.instances = instances if instances is not None else list()

        self.date = date
        self.Braak1 = Braak1
        self.Braak2 = Braak2
        self.Braak3 = Braak3
        self.Braak4 = Braak4
        self.Braak5 = Braak5
        self.Braak6 = Braak6
        self.Braak1to4 = Braak1to4
        self.Braak1_2 = Braak1_2
        self.Braak3_4 = Braak3_4
        self.Braak5_6 = Braak5_6
        self.MetaTemporal = MetaTemporal

        self.tracer = tracer
        self.patient_id = patient_id
        self.dataset = dataset
        self.age = age
        self.sex = sex
        self.apoe = apoe
        self.edu = edu
        self.mmse = mmse
        self.status = status

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.date = in_dict['date']
            self.Braak1 = in_dict['Braak1']
            self.Braak2 = in_dict['Braak2']
            self.Braak3 = in_dict['Braak3']
            self.Braak4 = in_dict['Braak4']
            self.Braak5 = in_dict['Braak5']
            self.Braak6 = in_dict['Braak6']
            self.Braak1to4 = in_dict['Braak1to4']
            self.Braak1_2 = in_dict.get('Braak1_2')
            self.Braak3_4 = in_dict.get('Braak3_4')
            self.Braak5_6 = in_dict.get('Braak5_6')
            self.MetaTemporal = in_dict.get('MetaTemporal')
            self.tracer = in_dict['tracer']
            self.patient_id = in_dict['patient_id']
            self.dataset = in_dict['dataset']
            self.age = in_dict['age']
            self.sex = in_dict['sex']
            self.apoe = in_dict['apoe']
            self.edu = in_dict['edu']
            self.mmse = in_dict['mmse']
            self.status = in_dict['status']

            for instance in in_dict['instances']:
                self.instances.append(DatasetInstance().from_dict(instance))

        return self
