#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import os
from kortex_driver.srv import *
from kortex_driver.msg import * # Base_JointSpeeds 등 사용

try:
    from mppi_solver import MPPIController
except ImportError:
    rospy.logerr("mppi_solver.py를 찾을 수 없습니다.")
    sys.exit()

class Gen3LiteMPPINode:
    def __init__(self):
        try:
            rospy.init_node('gen3_lite_mppi_integrated_node')
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.urdf_path = os.path.join(current_dir, "gen3_lite.urdf")
            
            self.mppi = MPPIController(self.urdf_path)
            self.nq = self.mppi.nq 

            self.q_curr = None
            self.is_init_success = False

            self.setup_services()
            self.action_topic_sub = rospy.Subscriber(f"/{self.robot_name}/action_topic", ActionNotification, self.cb_action_topic)
            self.sub_feedback = rospy.Subscriber(f"/{self.robot_name}/base_feedback", BaseCyclic_Feedback, self.cb_joint_feedback)
            
            # [수정 2] 'Ros Control' 대신 'Kinova Native Topic' 사용
            # 타입이 Float64MultiArray -> Base_JointSpeeds로 변경됨
            self.pub_vel = rospy.Publisher(f"/{self.robot_name}/in/joint_velocity", Base_JointSpeeds, queue_size=1)

            rospy.on_shutdown(self.stop_robot)
            self.is_init_success = True
            rospy.loginfo("✅ 시스템 초기화 완료")

        except Exception as e:
            rospy.logerr(f"초기화 중 오류 발생: {e}")

    def setup_services(self):
        prefix = f"/{self.robot_name}"
        services = {
            'clear_faults': (prefix + '/base/clear_faults', Base_ClearFaults),
            'set_ref_frame': (prefix + '/control_config/set_cartesian_reference_frame', SetCartesianReferenceFrame),
            'activate_notif': (prefix + '/base/activate_publishing_of_action_topic', OnNotificationActionTopic)
        }
        for name, (path, srv_type) in services.items():
            rospy.wait_for_service(path, timeout=5.0)
            setattr(self, name, rospy.ServiceProxy(path, srv_type))

    def cb_action_topic(self, msg):
        pass

    def cb_joint_feedback(self, msg):
        # Kinova는 Degree(도) 단위로 줌 -> Radian 변환
        q_arm = [msg.actuators[i].position for i in range(6)]
        q_full = np.zeros(self.nq)
        q_full[:6] = np.deg2rad(q_arm)
        q_full[6:] = 0.0
        self.q_curr = q_full

    def stop_robot(self):
        rospy.logwarn("⚠️ 로봇 정지 신호 전송")
        # 정지 시에도 Base_JointSpeeds 형식으로 보내야 함
        msg = Base_JointSpeeds()
        msg.joint_speeds = [JointSpeed(i, 0.0, 0) for i in range(6)]
        self.pub_vel.publish(msg)

    def prepare_hardware(self):
        rospy.loginfo("1. 결함(Faults) 제거...")
        self.clear_faults()
        frame_req = SetCartesianReferenceFrameRequest()
        frame_req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        self.set_ref_frame(frame_req)
        self.activate_notif(OnNotificationActionTopicRequest())
        rospy.sleep(1.0)
        return True

    def run_mppi_loop(self, target_P, target_R):
        rospy.loginfo("🚀 MPPI 제어 시작")
        
        # [수정 1] 제어 주기를 10Hz(0.1초)로 낮춤 -> 연산 지연 해결
        rate = rospy.Rate(10) 
        prev_dq = np.zeros(6)
        alpha = 0.6 

        while not rospy.is_shutdown():
            if self.q_curr is None: continue

            # 연산 시간 측정
            start_time = rospy.get_time()

            # 1. MPPI 계산
            u_opt = self.mppi.compute_action(self.q_curr, target_P, target_R)
            
            # 2. IK (관절 속도 변환)
            dq_full = self.mppi.dyn.solve_ik(self.q_curr, u_opt)
            dq_arm = dq_full[:6] 

            # 3. 필터링 및 클리핑
            dq_arm = alpha * prev_dq + (1 - alpha) * dq_arm
            dq_arm = np.clip(dq_arm, -0.2, 0.2) 
            prev_dq = dq_arm

            # --- [수정 3] 로봇에게 보낼 메시지 생성 (Native Format) ---
            # 계산된 Rad/s를 Kinova가 이해하는 Deg/s로 변환해야 함!
            dq_deg = np.rad2deg(dq_arm)
            
            msg = Base_JointSpeeds()
            msg.joint_speeds = []
            for i in range(6):
                js = JointSpeed()
                js.joint_identifier = i
                js.value = dq_deg[i] # 단위: 도/초 (deg/s)
                js.duration = 0.1     # 0이면 다음 명령 올 때까지 유지
                msg.joint_speeds.append(js)
            
            self.pub_vel.publish(msg)
            # ----------------------------------------------------

            # 4. 도착 판정
            _, curr_P, curr_R, _ = self.mppi.dyn.step(self.q_curr, np.zeros(6))
            pos_err = np.linalg.norm(curr_P - target_P)
            rot_err = 3.0 - np.trace(np.dot(target_R.T, curr_R))
            
            calc_time = rospy.get_time() - start_time
            if calc_time > 0.1: # 0.1초 넘으면 경고
                 rospy.logwarn_throttle(1, f"연산 지연: {calc_time:.3f}초")
                 # 연산 지연 시 안전 정지
                 stop_msg = Base_JointSpeeds()
                 stop_msg.joint_speeds = [JointSpeed(i, 0.0, 0) for i in range(6)]
                 self.pub_vel.publish(stop_msg)
                 prev_dq = np.zeros(6)  # 필터 초기화
                 rate.sleep()
                 continue

            if pos_err < 0.02 and rot_err < 0.1:
                # 정지 명령
                stop_msg = Base_JointSpeeds()
                stop_msg.joint_speeds = [JointSpeed(i, 0.0, 0) for i in range(6)]
                self.pub_vel.publish(stop_msg)
                
                dq_arm = np.zeros(6) # 필터 초기화용
                rospy.loginfo_throttle(2, f"✅ 목표 도달 유지 중 (오차: {pos_err:.3f}m)")

            rate.sleep()

    def main(self):
        if not self.is_init_success: return

        if self.prepare_hardware():
            while self.q_curr is None and not rospy.is_shutdown():
                rospy.sleep(0.1)
            
            _, start_P, start_R, _ = self.mppi.dyn.step(self.q_curr, np.zeros(self.nq))
            
            target_P = start_P.copy()
            target_P[2] += 0.05
            target_R = start_R.copy() 
            
            rospy.loginfo(f"📍 목표: 현재 높이 {start_P[2]:.3f}m -> {target_P[2]:.3f}m")
            self.run_mppi_loop(target_P, target_R)

if __name__ == "__main__":
    node = Gen3LiteMPPINode()
    node.main()
