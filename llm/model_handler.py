
# Not used 
class ModelHandler:



    def set_model(self, model):
        """Attach any backend model that implements send(message:str)->str."""
        self._model = model
