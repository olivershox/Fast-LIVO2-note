/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>
//子稀疏地图
// 定义一个名为 SubSparseMap 的结构体，用于表示子稀疏地图
struct SubSparseMap
{
  // 存储预测误差的向量
  vector<float> propa_errors;
  // 存储实际误差的向量
  vector<float> errors;
  // 存储仿射变换后的图像块的二维向量，每个元素是一个图像块的像素值向量
  vector<vector<float>> warp_patch;
  // 存储搜索层级的向量
  vector<int> search_levels;
  // 存储视觉点指针的向量
  vector<VisualPoint *> voxel_points;
  // 存储逆曝光时间列表的向量
  vector<double> inv_expo_list;
  // 存储从体素地图添加的带有方差的点的向量
  vector<pointWithVar> add_from_voxel_map;

  // 构造函数，预分配内存以提高性能
  SubSparseMap()
  {
    // 为 propa_errors 向量预分配 SIZE_LARGE 大小的内存
    propa_errors.reserve(SIZE_LARGE);
    // 为 errors 向量预分配 SIZE_LARGE 大小的内存
    errors.reserve(SIZE_LARGE);
    // 为 warp_patch 向量预分配 SIZE_LARGE 大小的内存
    warp_patch.reserve(SIZE_LARGE);
    // 为 search_levels 向量预分配 SIZE_LARGE 大小的内存
    search_levels.reserve(SIZE_LARGE);
    // 为 voxel_points 向量预分配 SIZE_LARGE 大小的内存
    voxel_points.reserve(SIZE_LARGE);
    // 为 inv_expo_list 向量预分配 SIZE_LARGE 大小的内存
    inv_expo_list.reserve(SIZE_LARGE);
    // 为 add_from_voxel_map 向量预分配 SIZE_SMALL 大小的内存
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  // 重置函数，用于清空所有存储的数据
  void reset()
  {
    // 清空 propa_errors 向量
    propa_errors.clear();
    // 清空 errors 向量
    errors.clear();
    // 清空 warp_patch 向量
    warp_patch.clear();
    // 清空 search_levels 向量
    search_levels.clear();
    // 清空 voxel_points 向量
    voxel_points.clear();
    // 清空 inv_expo_list 向量
    inv_expo_list.clear();
    // 清空 add_from_voxel_map 向量
    add_from_voxel_map.clear();
  }
};

// 定义一个名为 Warp 的类，用于表示仿射变换相关信息
class Warp
{
public:
  // 存储从当前帧到参考帧的仿射变换矩阵
  Matrix2d A_cur_ref;
  // 存储搜索层级
  int search_level;
  // 构造函数，初始化搜索层级和仿射变换矩阵
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  // 析构函数，当前为空，不执行额外操作
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

class VIOManager
{
public:
  int grid_size;
  vk::AbstractCamera *cam;
  vk::PinholeCamera *pinhole_cam;
  StatesGroup *state;
  StatesGroup *state_propagat;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;
  
  SubSparseMap *visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map; 
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void updateStateInverse(cv::Mat img, int level);
  void updateState(cv::Mat img, int level);
  void processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void computeJacobianAndUpdateEKF(cv::Mat img);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_