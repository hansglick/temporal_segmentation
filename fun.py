import numpy as np
import matplotlib.pyplot as plt




class Events (object):

    """
    Classe permettant de représenter l'ensemble des events desquels on veut extraire de l'information
    """


    def __init__(self,events_list=None):

        """
        - events_list : liste d'events sous la forme de UNIX EPOCH
        """

        if events_list is not None:
            self.EVENTS = np.array(events_list)
        else:
            self.EVENTS = None




    def Add_Simulation_Data(self,
        n_plages,
        n_events,
        amplitude,
        pollution):

        """
        - n_plages : nombre de plages avec fréquence différente
        - n_events : nombre d'events à simuler
        - amplitude : valeur de l'event le plus récent
        - pollution : float entre 0 et 1
        """

        # Intermediate Computations
        alpha = [1] * n_plages
        relative_markers = np.random.dirichlet(alpha)
        frequencies = np.random.uniform(low = 0, high = 10,size = n_plages)
        relative_weights = (relative_markers * frequencies) / np.dot(relative_markers,frequencies)
        n_events_by_plage = (relative_weights * n_events).astype(int)
        plage_sizes = (relative_markers * amplitude).astype(int)

        # Generation des événements
        x = np.cumsum(np.insert(relative_markers,0,0))
        bornes = ((np.vstack((x[:-1],x[1:])).T) * amplitude).astype(int)
        L = []
        for bo,n in zip(bornes,n_events_by_plage):
            a,b = bo
            events = np.linspace(a,b,n).astype(int)
            idx_to_remove = np.random.choice(np.arange(len(events)),replace = False,size=int(pollution*len(events)))
            events = np.delete(events,idx_to_remove)
            n_to_add = int(pollution*len(events))
            events_to_add = np.random.randint(a,b,n_to_add)
            events = np.hstack((events,events_to_add))
            L.append(events)


        final_results = np.hstack(L)

        self.EVENTS = final_results
        self.EVENTS_CLUSTERED = L
        self.FREQ = frequencies
        self.MARKERSR = relative_markers
        self.CLUSTERS = n_events_by_plage




    def Plot_Events(self,color=True):

        marks = self.Global_Frequence_Marks()
        LOC,BOOL = self.Plotted_Marks(marks)
        loctrue = LOC[BOOL]
        locfalse = LOC[~BOOL]
        

        if color:
            plt_1 = plt.figure(figsize=(30, 8))
            for cluster in self.EVENTS_CLUSTERED:
                y = np.repeat(0,len(cluster))
                plt.scatter(cluster,y)
            plt.ylim(-2, 2)
            plt.show()

        else:
            plt_1 = plt.figure(figsize=(30, 8))
            y = np.repeat(0,len(self.EVENTS))
            plt.scatter(self.EVENTS,y)

            y = np.repeat(1,len(loctrue))
            plt.scatter(loctrue,y,c="blue",s = 500,marker = "s")

            y = np.repeat(1,len(locfalse))
            plt.scatter(locfalse,y,c="grey",s = 500,marker = "s")

            y = np.repeat(0,len(marks))
            plt.scatter(marks,y,c="red",marker = "|",s = 500)

            plt.ylim(-1, 2)
            plt.show()



    def Print_Simulation_Data(self):
        
        print(self.CLUSTERS)
        print(self.FREQ)
        print(self.MARKERSR)





    def Global_Frequence_Marks(self):
        
        e = np.sort(self.EVENTS)
        lags = np.diff(e)
        maxlag_index = np.argmax(lags)
        maxlag_value = lags[maxlag_index]
        amplitude = (maxlag_value / 2)
        pivot_marker = e[maxlag_index] + amplitude
        b = np.arange(pivot_marker,e[-1]+amplitude,amplitude + 1)
        a = np.arange(pivot_marker,e[0]-amplitude,-amplitude - 1)
        a = a[1:]
        marks = np.hstack((a,b))
        marks = np.sort(marks)
        marks = marks.astype(int)
        
        return marks



    def Plotted_Marks(self,marks):
        
        BOOL = []
        LOC = []
        for idx,markinf in enumerate(marks[:-1]):
            marksup = marks[idx+1]
            a = self.EVENTS > markinf
            b = self.EVENTS < marksup
            res = np.sum(a & b) > 0
            loc = (marksup + markinf) / 2
            LOC.append(loc)
            BOOL.append(res)
        
        return np.array(LOC),np.array(BOOL)


