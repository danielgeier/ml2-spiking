TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    onecrossing_lanelet.cpp \
    lanelet_information.cpp

include(deployment.pri)
qtcAddDeployment()

