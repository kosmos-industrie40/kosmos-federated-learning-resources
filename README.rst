===================================
kosmos federated learning resources
===================================


DESCRIPTION
===========
This is the resource implementation of the kosmos federated learning framework. The whole project consists of two additional components `kosmos federated learning server <https://github.com/kosmos-industrie40/kosmos-federated-learning-server>`_ and `kosmos federated learning server <https://github.com/kosmos-industrie40/kosmos-federated-learning-server>`_ project. This project is able to run with any arbitrary data set but by default is executed with the bearing data set. For further information on design principals take a look at the `blogpost <https://www.inovex.de/de/blog/federated-learning-part-3/>`_ describing the whole project.

USE CASE
========
The general goal is to collect machine data from machine operators at the KOSMoS Edge and then collaboratively train a model for remaining useful lifetime prediction. This Federated Bearing use case implements this approach with the following restrictions:

- The data used for training isn't collected by the machine operator but the bearing data set manually distributed to the collaborating clients
- The current project can be deployed with the docker container provided in this project but isn't deployed in the current KOSMOS project
- The connection between clients and host isn't encrypted by now. This can be enabled in the Wrapper (websocket) and flower (grpc) implementation quite easy.

Open Points are:

- The optional but useful security features of Differential Privacy and Secure Multiparty Computation are not implemented yet.
