/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <unistd.h>

using namespace UNITREE_LEGGED_SDK;

#define SURFACE 0
#define EVAL_ID 0

class Custom
{
public:
    /*
    // Low-level port.
    Custom(uint8_t level): 
        udp(level, 8090, "192.168.123.10", 8007){
    }
    */
    // High-level port.
    Custom(uint8_t level): 
        udp(level, 8090, "192.168.123.220", 8082){
    }
    void UDPUpdate();
    void RobotControl();

    void UDPSend();
    void UDPRecv();

    UDP udp;
    //LowState state = {0};
    HighState state = {0};
    float dt = 0.002;     // 0.001~0.01

};

void Custom::UDPUpdate()
{ 
    udp.Recv();
    udp.Send();
}

void Custom::RobotControl() 
{
    // Timestamp.
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    // Receive state data.
    udp.GetRecv(state);

    // Output all data as a CSV to stdout.
    std::cout << timestamp << "," << SURFACE << ",";
    std::cout << state.velocity[0] << ",";
    std::cout << state.velocity[1] << ",";
    std::cout << state.velocity[2] << ",";
    std::cout << state.yawSpeed << ",";

    std::cout << EVAL_ID << ",";

    for (uint8_t i = 0; i < 4; i++) {
        std::cout << state.motorState[i].tauEst << ",";
        std::cout << state.motorState[i+1].tauEst << ",";
        std::cout << state.motorState[i+2].tauEst << ",";
    }

    std::cout << state.imu.accelerometer[0] << ",";
    std::cout 	    << state.imu.accelerometer[1] << ",";
    std::cout 	    << state.imu.accelerometer[2] << ",";
    std::cout 	    << state.imu.gyroscope[0] << ",";
    std::cout 	    << state.imu.gyroscope[1] << ",";
    std::cout 	    << state.imu.gyroscope[2] << ",";
    std::cout 	    << state.imu.quaternion[0] << ",";
    std::cout 	    << state.imu.quaternion[1] << ",";
    std::cout 	    << state.imu.quaternion[2] << ",";
    std::cout 	    << state.imu.quaternion[3];

    std::cout << std::endl;

}

int main(void)
{
    //std::cout << "Communication level is set to LOW-level." << std::endl
    //          << "WARNING: Make sure the robot is hung up." << std::endl
    //         << "Press Enter to continue..." << std::endl;
    //std::cin.ignore();
    // Print headers.
    std::cout << "time (ms),surface,forwardSpeed (m/s),sideSpeed (m/s),rotateSpeed (m?/s),yawSpeed(rad/s),eval_id,frHip t (Nm),FLHipT (Nm),RRHipT (Nm),RLHipT (Nm),FRThighT (Nm),FLThighT (Nm),RRThighT (Nm),RLThighT (Nm),FRCalfT (Nm),FLCalfT (Nm),RRCalfT (Nm),RLCalfT (Nm),IMUAccx,IMUAccy,IMUAccz,IMUGyrroll,IMUGyrpitch,IMUGyryaw,IMUQw,IMUQx,IMUQy,IMUQz\n";

    Custom custom(HIGHLEVEL);
    InitEnvironment();
    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udp("udp_update",     custom.dt, 3, boost::bind(&Custom::UDPUpdate,      &custom));

    loop_control.start();
    loop_udp.start();

    while(1){
        sleep(10);
    };

    return 0; 
}
