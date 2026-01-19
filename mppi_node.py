#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import os
from std_msgs.msg import Float64MultiArray
from kortex_driver.srv import *
from kortex_driver.msg import *

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
            
            # [수정 1] Pinocchio 의존성 제거 (mppi 객체에서 직접 가져옴)
            self.nq = self.mppi.nq 

            self.q_curr = None
            self.is_init_success = False

            self.setup_services()
            self.action_topic_sub = rospy.Subscriber(f"/{self.robot_name}/action_topic", ActionNotification, self.cb_action_topic)
            self.sub_feedback = rospy.Subscriber(f"/{self.robot_name}/base_feedback", BaseCyclic_Feedback, self.cb_joint_feedback)
            self.pub_vel = rospy.Publisher(f"/{self.robot_name}/joint_group_velocity_controller/command", Float64MultiArray, queue_size=1)

            rospy.on_shutdown(self.stop_robot)
            self.is_init_success = True
            rospy.loginfo("✅ 시스템 초기화 완료")

        except Exception as e:
            rospy.logerr(f"초기화 중 오류 발생: {e}")

    def setup_services(self):
        # (기존과 동일)
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
        # (기존과 동일)
        q_arm = [msg.actuators[i].position for i in range(6)]
        q_full = np.zeros(self.nq)
        q_full[:6] = np.deg2rad(q_arm)
        q_full[6:] = 0.0
        self.q_curr = q_full

    def stop_robot(self):
        rospy.logwarn("⚠️ 로봇 정지")
        msg = Float64MultiArray(data=[0.0] * 6)
        self.pub_vel.publish(msg)

    def prepare_hardware(self):
        # (기존과 동일)
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
        rate = rospy.Rate(50) 
        prev_dq = np.zeros(6)
        alpha = 0.6 

        while not rospy.is_shutdown():
            if self.q_curr is None: continue

            # [수정 2] 함수 이름 변경 (get_optimal_command -> compute_action)
            # MPPI가 계산한 값은 이미 '관절 속도(Joint Velocity)'입니다.
            u_opt = self.mppi.compute_action(self.q_curr, target_P) # target_R은 solver에서 안씀
            
            # [수정 3] 불필요한 IK 제거
            # u_opt가 이미 최적 관절 속도이므로 solve_ik를 통과시키지 않습니다.
            dq_arm = u_opt[:6] # 전체 관절 중 팔 부분(6개)만 추출

            # 3. 속도 필터링 및 안전 클램핑
            dq_arm = alpha * prev_dq + (1 - alpha) * dq_arm
            
            # [중요] dynamics.py의 0.2 제한과 맞추거나 더 보수적으로 설정
            dq_arm = np.clip(dq_arm, -0.2, 0.2) 
            prev_dq = dq_arm

            # 4. 도착 판정
            _, curr_P, curr_R, _ = self.mppi.dyn.step(self.q_curr, np.zeros(self.nq))
            dist = np.linalg.norm(curr_P - target_P)
            
            if dist < 0.02:
                dq_arm = np.zeros(6)
                rospy.loginfo_throttle(2, "목표 도달 (유지 중...)")

            # 5. 명령 발행
            msg = Float64MultiArray(data=dq_arm.tolist())
            self.pub_vel.publish(msg)
            
            rate.sleep()

    def main(self):
        if not self.is_init_success: return

        if self.prepare_hardware():
            while self.q_curr is None and not rospy.is_shutdown():
                rospy.sleep(0.1)
            
            # 현위치 기반 목표 설정
            # step 함수 호출 시 인자 개수 맞춤 (속도 0 벡터)
            _, start_P, start_R, _ = self.mppi.dyn.step(self.q_curr, np.zeros(self.nq))
            
            target_P = start_P.copy()
            target_P[2] += 0.10  # 10cm 상승
            target_R = start_R.copy() 
            
            rospy.loginfo(f"📍 목표: 현재 높이 {start_P[2]:.3f}m -> {target_P[2]:.3f}m")
            self.run_mppi_loop(target_P, target_R)

if __name__ == "__main__":
    node = Gen3LiteMPPINode()
    node.main()