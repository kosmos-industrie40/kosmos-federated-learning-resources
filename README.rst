===================================
KOSMoS Federated Learning Resources
===================================


DESCRIPTION
===========
This repository contains shared resources needed for the KOSMoS Federated Learning Framework.
The framework consists of two additional components: the `KOSMoS Federated Learning Client <https://github.com/kosmos-industrie40/kosmos-federated-learning-client>`_ and the `KOSMoS Federated Learning Server <https://github.com/kosmos-industrie40/kosmos-federated-learning-server>`_.
This project is able to run with any arbitrary data set but by default is executed with the bearing data set.
For further information on design principals take a look at the `blogpost <https://www.inovex.de/de/blog/federated-learning-implementation-into-kosmos-part-3/>`_ describing the whole project.

USE CASE
========
The general goal is to collect machine data from machine operators at the KOSMoS Edge and then collaboratively train a model to predict the remaining useful lifetime. This Federated Bearing use case implements this approach with the following restrictions:

- The data used for training is not collected by the machine operator but the bearing data set is manually distributed to the collaborating clients
- The current project can be deployed with the docker container provided in this project but isn't deployed in the current KOSMoS project
- The connection between clients and host isn't encrypted as of now. This can be enabled in the Wrapper (websocket) and flower (grpc) implementation quite easily.

Open Points are:

- The optional but useful security features Differential Privacy and Secure Multiparty Computation are not implemented yet.


DATASET
=======
As default dataset a modified version of the FEMTO-ST Bearings Dataset [NECT2012]_ is used. See `DATASETCHANGES.rst <./DATASETCHANGES.rst>`_ for further information on the modifications.


Note
====

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.

References
==========

.. [NECT2012] Patrick Nectoux, Rafael Gouriveau, Kamal Medjaher, Emmanuel Ramasso, Brigitte Morello, Noureddine Zerhouni, Christophe Varnier. (2012). PRONOSTIA : An experimental platform for bearings accelerated degradation tests. In IEEE International Conference on Prognostics and Health Management (pp. 1–8). Denver.
