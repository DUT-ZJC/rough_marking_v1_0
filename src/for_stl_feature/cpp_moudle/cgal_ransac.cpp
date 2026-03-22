#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/property_map.h>
#include <CGAL/Random.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>

#include <vector>
#include <cmath>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef std::pair<Kernel::Point_3, Kernel::Vector_3>         Point_with_normal;
typedef std::vector<Point_with_normal>                       Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;
typedef CGAL::Shape_detection::Efficient_RANSAC_traits<Kernel, Pwn_vector, Point_map, Normal_map> Traits;
typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> Efficient_ransac;
typedef CGAL::Shape_detection::Plane<Traits>            Plane;
typedef CGAL::Shape_detection::Cylinder<Traits>         Cylinder;

namespace py = pybind11;

py::tuple extract_cgal(
    // 强制声明：只接受 C 连续内存的 double 型二维矩阵！不符合立刻报错，拒绝静默读错！
    py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<double, py::array::c_style | py::array::forcecast> normals,
    double probability,
    int min_points,
    double epsilon,
    double cluster_epsilon,
    double normal_threshold,
    int random_seed)
{
    // 使用最安全的 2D 数组访问器，拒绝一切裸指针！
    auto v_acc = vertices.unchecked<2>();
    auto n_acc = normals.unchecked<2>();
    
    size_t num_pts = vertices.shape(0);
    
    Pwn_vector points;
    points.reserve(num_pts);
    for(size_t i = 0; i < num_pts; ++i) {
        // v_acc(i, 0) 绝对安全地读取第 i 个点的 X 坐标
        points.emplace_back(
            Kernel::Point_3(v_acc(i, 0), v_acc(i, 1), v_acc(i, 2)),
            Kernel::Vector_3(n_acc(i, 0), n_acc(i, 1), n_acc(i, 2))
        );
    }

    Efficient_ransac ransac;
    if (random_seed >= 0) {
        CGAL::get_default_random() = CGAL::Random(static_cast<unsigned int>(random_seed));
    }
    ransac.set_input(points);
    ransac.add_shape_factory<Plane>();
    ransac.add_shape_factory<Cylinder>();

    Efficient_ransac::Parameters parameters;
    parameters.probability = probability;
    parameters.min_points = static_cast<std::size_t>(min_points);
    parameters.epsilon = epsilon;
    parameters.cluster_epsilon = cluster_epsilon;
    parameters.normal_threshold = normal_threshold;

    ransac.detect(parameters);

    std::vector<std::vector<int>> planes_idx;
    std::vector<std::vector<double>> planes_param;
    std::vector<std::vector<int>> cyls_idx;
    std::vector<std::vector<double>> cyls_param;

    for(auto shape = ransac.shapes().begin(); shape != ransac.shapes().end(); ++shape) {
        if (Plane* plane = dynamic_cast<Plane*>(shape->get())) {
            std::vector<int> idx;
            double cx = 0.0, cy = 0.0, cz = 0.0;
            
            for(auto it = plane->indices_of_assigned_points().begin(); it != plane->indices_of_assigned_points().end(); ++it) {
                size_t pt_idx = *it;
                idx.push_back(static_cast<int>(pt_idx));
                cx += v_acc(pt_idx, 0);
                cy += v_acc(pt_idx, 1);
                cz += v_acc(pt_idx, 2);
            }
            
            if(!idx.empty()) {
                cx /= idx.size();
                cy /= idx.size();
                cz /= idx.size();
            }
            
            planes_idx.push_back(idx);
            
            Kernel::Vector_3 n = plane->plane_normal();
            double d = -(n.x() * cx + n.y() * cy + n.z() * cz);
            
            planes_param.push_back({n.x(), n.y(), n.z(), d, cx, cy, cz});
        }
        else if (Cylinder* cyl = dynamic_cast<Cylinder*>(shape->get())) {
            std::vector<int> idx;
            for(auto it = cyl->indices_of_assigned_points().begin(); it != cyl->indices_of_assigned_points().end(); ++it) {
                idx.push_back(static_cast<int>(*it));
            }
            cyls_idx.push_back(idx);

            Kernel::Line_3 axis = cyl->axis();
            Kernel::Point_3 p = axis.point(0);
            Kernel::Vector_3 dir = axis.to_vector();
            double len = std::sqrt(dir.x()*dir.x() + dir.y()*dir.y() + dir.z()*dir.z());
            cyls_param.push_back({p.x(), p.y(), p.z(), dir.x()/len, dir.y()/len, dir.z()/len, cyl->radius()});
        }
    }

    return py::make_tuple(planes_idx, planes_param, cyls_idx, cyls_param);
}

PYBIND11_MODULE(cgal_ransac, m) {
    m.def(
        "extract_shapes",
        &extract_cgal,
        py::arg("vertices"),
        py::arg("normals"),
        py::arg("probability"),
        py::arg("min_points"),
        py::arg("epsilon"),
        py::arg("cluster_epsilon"),
        py::arg("normal_threshold"),
        py::arg("random_seed") = -1
    );
}
