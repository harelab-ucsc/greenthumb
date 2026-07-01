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

    Safety safe;
    int motiontime = 0;
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
    motiontime += 2;
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    // Receive state data.
    udp.GetRecv(state);

    // Initialize motion commands:
    // * cmd.mode
    //      0:idle, default stand
    //      1:forced stand
    //      2:walk continuously
    cmd.mode = 0;
    cmd.gaitType = 0;
    cmd.speedLevel = 0;
    cmd.footRaiseHeight = 0;
    cmd.bodyHeight = 0;
    cmd.euler[0] = 0;
    cmd.euler[1] = 0;
    cmd.euler[2] = 0;
    cmd.velocity[0] = 0.0f;
    cmd.velocity[1] = 0.0f;
    cmd.yawSpeed = 0.0f;
    cmd.reserve = 0;

    if (motiontime > 0)
    {
        cmd.mode = 2;
    }

    // Output all data as a CSV to stdout.
    std::cout << timestamp << "," << SURFACE << ",";
    std::cout << state.velocity[0] << ",";
    std::cout << state.velocity[1] << ",";
    std::cout << state.velocity[2] << ",";
    std::cout << state.yawSpeed << ",";

    std::cout << EVAL_ID << ",";

    // Angle of each joint.
    for (uint8_t i = 0; i < 4; i++) {
        std::cout << state.motorState[i*3].q << ",";
        std::cout << state.motorState[i*3+1].q << ",";
        std::cout << state.motorState[i*3+2].q << ",";
    }

    // Raw angle of each joint.
    for (uint8_t i = 0; i < 4; i++) {
        std::cout << state.motorState[i*3].q_raw << ",";
        std::cout << state.motorState[i*3+1].q_raw << ",";
        std::cout << state.motorState[i*3+2].q_raw << ",";
    }

    // Velocity of each joint (angular).
    for (uint8_t i = 0; i < 4; i++) {
        std::cout << state.motorState[i*3].dq << ",";
        std::cout << state.motorState[i*3+1].dq << ",";
        std::cout << state.motorState[i*3+2].dq << ",";
    }

    // Raw velocity of each joint (angular).
    for (uint8_t i = 0; i < 4; i++) {
        std::cout << state.motorState[i*3].dq_raw << ",";
        std::cout << state.motorState[i*3+1].dq_raw << ",";
        std::cout << state.motorState[i*3+2].dq_raw << ",";
    }

    // Acceleration of each joint (angular).
    for (uint8_t i = 0; i < 4; i++) {
        std::cout << state.motorState[i*3].ddq << ",";
        std::cout << state.motorState[i*3+1].ddq << ",";
        std::cout << state.motorState[i*3+2].ddq << ",";
    }

    // Raw acceleration of each joint (angular).
    for (uint8_t i = 0; i < 4; i++) {
        std::cout << state.motorState[i*3].ddq_raw << ",";
        std::cout << state.motorState[i*3+1].ddq_raw << ",";
        std::cout << state.motorState[i*3+2].ddq_raw << ",";
    }

    // Torque for each joint.
    for (uint8_t i = 0; i < 4; i++) {
        std::cout << state.motorState[i*3].tauEst << ",";
        std::cout << state.motorState[i*3+1].tauEst << ",";
        std::cout << state.motorState[i*3+2].tauEst << ",";
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

    // Send control command.
    udp.SetSend(cmd);
}

int main(void)
{
    //std::cout << "Communication level is set to LOW-level." << std::endl
    //          << "WARNING: Make sure the robot is hung up." << std::endl
    //         << "Press Enter to continue..." << std::endl;
    //std::cin.ignore();
    // Print headers.
    std::cout << "time (ms),surface,forwardSpeed (m/s),sideSpeed (m/s),rotateSpeed (m?/s),yawSpeed(rad/s),eval_id,";

    // Joint angles.
    std::cout << "FRHipQ (rad),FRThighQ (rad),FRKneeQ (rad),";
    std::cout << "FLHipQ (rad),FLThighQ (rad),FLKneeQ (rad),";
    std::cout << "RRHipQ (rad),RRThighQ (rad),RRKneeQ (rad),";
    std::cout << "RLHipQ (rad),RLThighQ (rad),RLKneeQ (rad),";

    // Raw joint angles.
    std::cout << "FRHipQ (raw rad),FRThighQ (raw rad),FRKneeQ (raw rad),";
    std::cout << "FLHipQ (raw rad),FLThighQ (raw rad),FLKneeQ (raw rad),";
    std::cout << "RRHipQ (raw rad),RRThighQ (raw rad),RRKneeQ (raw rad),";
    std::cout << "RLHipQ (raw rad),RLThighQ (raw rad),RLKneeQ (raw rad),";

    // Joint velocities.
    std::cout << "FRHipdQ (rps),FRThighdQ (rps),FRKneedQ (rps),";
    std::cout << "FLHipdQ (rps),FLThighdQ (rps),FLKneedQ (rps),";
    std::cout << "RRHipdQ (rps),RRThighdQ (rps),RRKneedQ (rps),";
    std::cout << "RLHipdQ (rps),RLThighdQ (rps),RLKneedQ (rps),";

    // Raw joint velocities.
    std::cout << "FRHipdQ (raw rps),FRThighdQ (raw rps),FRKneedQ (raw rps),";
    std::cout << "FLHipdQ (raw rps),FLThighdQ (raw rps),FLKneedQ (raw rps),";
    std::cout << "RRHipdQ (raw rps),RRThighdQ (raw rps),RRKneedQ (raw rps),";
    std::cout << "RLHipdQ (raw rps),RLThighdQ (raw rps),RLKneedQ (raw rps),";

    // Joint accelerations.
    std::cout << "FRHipd2Q (rps^2),FRThighd2Q (rps^2),FRKneed2Q (rps^2),";
    std::cout << "FLHipd2Q (rps^2),FLThighd2Q (rps^2),FLKneed2Q (rps^2),";
    std::cout << "RRHipd2Q (rps^2),RRThighd2Q (rps^2),RRKneed2Q (rps^2),";
    std::cout << "RLHipd2Q (rps^2),RLThighd2Q (rps^2),RLKneed2Q (rps^2),";

    // Raw joint accelerations.
    std::cout << "FRHipd2Q (raw rps^2),FRThighd2Q (raw rps^2),FRKneed2Q (raw rps^2),";
    std::cout << "FLHipd2Q (raw rps^2),FLThighd2Q (raw rps^2),FLKneed2Q (raw rps^2),";
    std::cout << "RRHipd2Q (raw rps^2),RRThighd2Q (raw rps^2),RRKneed2Q (raw rps^2),";
    std::cout << "RLHipd2Q (raw rps^2),RLThighd2Q (raw rps^2),RLKneed2Q (raw rps^2),";

    // Estimated joint torques.
    std::cout << "FRHipT (Nm),FRHipT (Nm),FRHipT (Nm),FRHipT (Nm),";
    std::cout << "FRThighT (Nm),FRThighT (Nm),FRThighT (Nm),FRThighT (Nm),";
    std::cout << "FRKneeT (Nm),FRKneeT (Nm),FRKneeT (Nm),FRKneeT (Nm),";

    // IMU out.
    std::cout << "IMUAccx,IMUAccy,IMUAccz,IMUGyrroll,IMUGyrpitch,IMUGyryaw,IMUQw,IMUQx,IMUQy,IMUQz\n";

    Custom custom(HIGHLEVEL);
    InitEnvironment();
    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
<<<<<<< HEAD
=======
    LoopFunc loop_udpSend("udp_send", custom.dt, 3, boost::bind(&Custom::UDPSend, &custom));
    LoopFunc loop_udpRecv("udp_recv", custom.dt, 3, boost::bind(&Custom::UDPRecv, &custom));
>>>>>>> main
    LoopFunc loop_udp("udp_update",     custom.dt, 3, boost::bind(&Custom::UDPUpdate,      &custom));

    loop_control.start();
    loop_udp.start();

    while(1){
        sleep(10);
    };

    return 0; 
}
