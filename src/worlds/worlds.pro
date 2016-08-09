TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    onecrossing_lanelet.cpp \
    lanelet_information.cpp \
    lanelet_random_pos.cpp \
    lanelet_utils.cpp

include(deployment.pri)
qtcAddDeployment()

OTHER_FILES += \
    catkin-build.sh \
    onecrossing.launch \
    tcrossing.launch \
    random_pos_service.srv

HEADERS += \
    lanelet_information.h \
    lanelet_random_pos.h \
    lanelet_utils.h

