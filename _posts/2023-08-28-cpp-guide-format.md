---
layout: articles
title: C++最佳实践-格式format相关
tags: c++
---

## 行长度相关
### 代码行宽度推荐80不允许超过120

[其中80主要考虑的是人眼视线范围而非屏宽](https://katafrakt.me/2017/09/16/80-characters-line-length-limit/)
## 非法ASCII 字符相关
代码中不要添加非ASCII的字符，允许中文注释
## 空格和制表位相关
缺省缩进为 2 个空格.使用空格而不是tab
## 函数声明和定义格式相关
### 对显式重写的虚函数要使用override修饰。重写虚函数时不要添加virtual关键字

```cpp
class Base {
 public:
  virtual void do_something() {}
};

class Derived ：public Base {
 public:
  void do_something() override {}          // good
  virtual void do_something() override {}  // bad: remove virtual
};
```

### 所有形参应尽可能对齐，如果第一个参数无法和函数名放在同一行，则换行后的参数保持 4 个空格的缩进

```cpp
ReturnType LongClassName::ReallyReallyReallyLongFunctionName(
    Type par_name1,  // 4 space indent
    Type par_name2, Type par_name3) {
  DoSomething();  // 2 space indent
  ...
}
```
### 只有在参数未被使用或者其用途非常明显时,才省略参数名
Foo类的拷贝构造函数和赋值运算符重载函数被声明为删除函数（使用= delete），表示禁止进行拷贝构造和赋值操作。在这种情况下，参数名可以被省略，因为它们未被使用，且函数的用途非常明显

```cpp
class Foo {
 public:
  Foo(const Foo&) = delete;
  Foo& operator=(const Foo&) = delete;
};

```

### 如果返回类型和函数名在一行放不下，分行

```cpp
int mappages(pagetable_t pagetable, uint64 va, uint64 size, uint64 pa,
             int perm) {}
```


### 如果返回类型与函数声明或定义分行了, 不要缩进
```cpp
int
add(int a, int b)
{
    return a + b;
}
```

### 左圆括号总是和函数名在同一行
```cpp
int AVeryVeryVeryVeryLongFunctionName(pagetable_t pagetable,
                                      uint64 va,  // 4 spaces
                                      uint64 size, uint64 pa, int perm) {
  // do something
}
```

### 函数名和左圆括号间永远没有空格
```cpp
int FunctionName() {
  // do something
}
```

### 圆括号与参数间没有空格

```cpp
int FunctionName(pagetable_t pagetable) {
  // do something
}
```
### 左大括号总在最后一个参数同一行的末尾处，不另起新行, 右大括号总是单独位于函数最后一行, 或者与左大括号同一行
```cpp
int FunctionName(
    pagetable_t pagetable, uint64 va,
    uint64 size, uint64 pa, int perm) {
  // do something
}
```


### 右圆括号和左大括号间总是有一个空格

```cpp
int FunctionName(
    pagetable_t pagetable, uint64 va,
    uint64 size, uint64 pa, int perm) {
  do something();
}
```
## Lambda表达式格式相关

### 参数的格式和普通函数相同
```cpp
auto pow_n = [](int n) -> int { return n * n; }
```
### 引用捕获时，&和变量名之间不应有空格
```cpp
int x = 0;
auto x_plus_n = [&x](int n) -> int { return x + n; }
```
### 短lambda可以内联编写为函数参数，这时候也需要遵守函数参数的格式要求
```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5};

// 使用内联lambda函数作为函数参数
std::for_each(numbers.begin(), numbers.end(), [](int num) {
    std::cout << num << " ";
});
```

## 浮点字面量相关

### 浮点字面量应该始终有一个基数，两边都有数字，即使它们使用指数表示法

```cpp
float f = 1.0f;
float f2 = 1;   // Also OK
long double ld = -0.5L;
double d = 1248.0e6;
```
而不是
```cpp
float f = 1.f;
long double ld = -.5L;
double d = 1248e6;
```
## 条件相关
### 倾向于不在圆括号内使用空格。关键字 if 和 else 另起一行
```cpp
if (condition) {
  DoOneThing();
} else {
  DoNothing();
}
```
### 有 else if 时，最后一分分支必须 是 else


### 单行条件语句或者循环也需要使用大括号

### switch 语句中如果有不满足 case 条件的枚举值, switch 应该总是包含一个 default 匹配

如果有输入值没有 case 去处理, 编译器将给出 warning

如果 default 应该永远执行不到, 简单的加条 assert：

```cpp
switch (var) {
  case 0: {  // 2 space indent
    ...      // 4 space indent
    break;
  }
  case 1: {
    ...
    break;
  }
  default: {
    assert(false);
  }
}
```
## 循环相关
### 空循环体应使用 {} 或 continue, 而不是一个简单的分号
```cpp
while (condition) {
  // do something
}  // good

while (condition) continue;  // good

while (condition) {}  // good

while (condition) ;  // bad
```
## 指针和引用表达式格式相关
### 指针或引用表达式中，句点或箭头前后不要有空格. 指针/地址操作符 (*, &) 之后不能有空格。星号和引用符号与变量名紧挨

```C++
x = *p;
p = &x;
x = r.y;
x = r->y;

// These are fine, space preceding.
char *c;
const std::string &str;
int *GetPointer();
std::vector<char *>
```
而不是
```c++
char* c;
const std::string& str;
int* GetPointer();
std::vector<char*>  // Note no space between '*' and '>'
```
### 多重声明中不允许声明指针和引用 
```cpp
int *x, y;    // 不允许
```

## 布尔表达式格式相关
### 布尔表达式断行时，&&操作符必须位于行尾
```cpp
if (this_one_thing > this_other_thing &&
    a_third_thing == a_fourth_thing &&
    yet_another && last_one) {
  // ...
}
```
### 有时参数形成的结构对可读性很重要。在这些情况下，可以根据该结构自由设置参数格式
```cpp
// Transform the widget by a 3x3 matrix. 形象的3x3
my_widget.Transform(x1, x2, x3,
                    y1, y2, y3,
                    z1, z2, z3);
```
## 预处理指令格式相关
### 预处理指令不要缩进, 从行首开始 
```cpp
// Good - directives at beginning of line
  if (lopsided_score) {
#if DISASTER_PENDING      // Correct -- Starts at beginning of line
    DropEverything();
# if NOTIFY               // OK but not required -- Spaces after #
    NotifyClient();
# endif
#endif
    BackToNormal();
  }
```

## 类格式相关
### 访问控制块的声明依次序是 public:, protected:, private:, 每个都缩进 1 个空格。public 放在最前面, 然后是 protected, 最后是 private
## 命名空间格式相关
### 命名空间内容不缩进
```cpp
```C++
namespace {

void foo() {  // Correct.  No extra indentation within namespace.
  // ...
}

}  // namespace
```

### 命名空间结束需要添加注释注明所属的命名空间

### 不要加入无用的空行。可以用空行划分代码逻辑段
## 变量和数组初始化格式相关
### 使用 int x{3} 而不是 int x{ 3 }

```cpp
int x = 3;
int x(3);
int x{3};
std::string name = "Some Name";
std::string name("Some Name");
std::string name{"Some Name"};
```

使用{}进行初始化将不支持隐式转换：
```c++
int pi(3.14);  // OK -- pi == 3.
int pi{3.14};  // Compile error: narrowing conversion.
```
### 构造函数初始化列表，下面几个方式都可以

```cpp
// When everything fits on one line:
MyClass::MyClass(int var) : some_var_(var) {
  DoSomething();
}

// If the signature and initializer list are not all on one line,
// you must wrap before the colon and indent 4 spaces:
MyClass::MyClass(int var)
    : some_var_(var), some_other_var_(var + 1) {
  DoSomething();
}

// When the list spans multiple lines, put each member on its own line
// and align them:
MyClass::MyClass(int var)
    : some_var_(var),             // 4 space indent
      some_other_var_(var + 1) {  // lined up
  DoSomething();
}

// As with any other code block, the close curly can be on the same
// line as the open curly, if it fits.
MyClass::MyClass(int var)
    : some_var_(var) {}
```

## 操作符格式相关
### 建议每个二元操作符间添加空格

```cpp
// Assignment operators always have spaces around them.
x = 0;

// Other binary operators usually have spaces around them, but it's
// OK to remove spaces around factors.  Parentheses should have no
// internal padding.
v = w * x + y / z;    // good
v = w*x + y/z;        // bad
v = w * (x + z);      // good

// No spaces separating unary operators and their arguments.
x = -5;
++x;
if (x && !y)
  ...
```
## 整体例子

ros
```cpp
// Copyright 2014 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RCLCPP__NODE_IMPL_HPP_
#define RCLCPP__NODE_IMPL_HPP_

#include <rmw/error_handling.h>
#include <rmw/rmw.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rcl/publisher.h"
#include "rcl/subscription.h"
#include "rclcpp/contexts/default_context.hpp"
#include "rclcpp/create_client.hpp"
#include "rclcpp/create_generic_publisher.hpp"
#include "rclcpp/create_generic_subscription.hpp"
#include "rclcpp/create_publisher.hpp"
#include "rclcpp/create_service.hpp"
#include "rclcpp/create_subscription.hpp"
#include "rclcpp/create_timer.hpp"
#include "rclcpp/detail/resolve_enable_topic_statistics.hpp"
#include "rclcpp/parameter.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/timer.hpp"
#include "rclcpp/type_support_decl.hpp"
#include "rclcpp/visibility_control.hpp"

#ifndef RCLCPP__NODE_HPP_
#include "node.hpp"
#endif

namespace rclcpp {

/// Node is the single point of entry for creating publishers and subscribers.
class Node : public std::enable_shared_from_this<Node> {
 public:
  using OnSetParametersCallbackHandle =
      rclcpp::node_interfaces::OnSetParametersCallbackHandle;
  using OnParametersSetCallbackType = rclcpp::node_interfaces::
      NodeParametersInterface::OnParametersSetCallbackType;

  RCLCPP_SMART_PTR_DEFINITIONS(Node)

  /// Create a new node with the specified name.
  /**
   * \param[in] node_name Name of the node.
   * \param[in] options Additional options to control creation of the node.
   * \throws InvalidNamespaceError if the namespace is invalid
   */
  RCLCPP_PUBLIC
  explicit Node(const std::string &node_name,
                const NodeOptions &options = NodeOptions());

  /// Create a new node with the specified name.
  /**
   * \param[in] node_name Name of the node.
   * \param[in] namespace_ Namespace of the node.
   * \param[in] options Additional options to control creation of the node.
   * \throws InvalidNamespaceError if the namespace is invalid
   */
  RCLCPP_PUBLIC
  explicit Node(const std::string &node_name, const std::string &namespace_,
                const NodeOptions &options = NodeOptions());

  RCLCPP_PUBLIC
  virtual ~Node();

  /// Get the name of the node.
  /** \return The name of the node. */
  RCLCPP_PUBLIC
  const char *GetName() const;

  /// Get the namespace of the node.
  /**
   * This namespace is the "node's" namespace, and therefore is not affected
   * by any sub-namespace's that may affect entities created with this instance.
   * Use get_effective_namespace() to get the full namespace used by entities.
   *
   * \sa get_sub_namespace()
   * \sa get_effective_namespace()
   * \return The namespace of the node.
   */
  RCLCPP_PUBLIC
  const char *GetNamespace() const;

  /// Get the fully-qualified name of the node.
  /**
   * The fully-qualified name includes the local namespace and name of the node.
   * \return fully-qualified name of the node.
   */
  RCLCPP_PUBLIC
  const char *GetFullyQualifiedName() const;

  /// Get the logger of the node.
  /** \return The logger of the node. */
  RCLCPP_PUBLIC
  rclcpp::Logger GetLogger() const;

  /// Create and return a callback group.
  RCLCPP_PUBLIC
  rclcpp::CallbackGroup::SharedPtr CreateCallbackGroup(
      rclcpp::CallbackGroupType group_type,
      bool automatically_add_to_executor_with_node = true);

  /// Return the list of callback groups in the node.
  RCLCPP_PUBLIC
  const std::vector<rclcpp::CallbackGroup::WeakPtr> &GetCallbackGroups() const;

  /// Create and return a Publisher.
  /**
   * The rclcpp::QoS has several convenient constructors, including a
   * conversion constructor for size_t, which mimics older API's that
   * allows just a string and size_t to create a publisher.
   *
   * For example, all of these cases will work:
   *
   * ```cpp
   * pub = node->create_publisher<MsgT>("chatter", 10);  // implicitly KeepLast
   * pub = node->create_publisher<MsgT>("chatter", QoS(10));  // implicitly
   * KeepLast pub = node->create_publisher<MsgT>("chatter", QoS(KeepLast(10)));
   * pub = node->create_publisher<MsgT>("chatter", QoS(KeepAll()));
   * pub = node->create_publisher<MsgT>("chatter",
   * QoS(1).best_effort().durability_volatile());
   * {
   *   rclcpp::QoS custom_qos(KeepLast(10), rmw_qos_profile_sensor_data);
   *   pub = node->create_publisher<MsgT>("chatter", custom_qos);
   * }
   * ```
   *
   * The publisher options may optionally be passed as the third argument for
   * any of the above cases.
   *
   * \param[in] topic_name The topic for this publisher to publish on.
   * \param[in] qos The Quality of Service settings for the publisher.
   * \param[in] options Additional options for the created Publisher.
   * \return Shared pointer to the created publisher.
   */
  template <typename MessageT, typename AllocatorT = std::allocator<void>,
            typename PublisherT = rclcpp::Publisher<MessageT, AllocatorT>>
  std::shared_ptr<PublisherT> CreatePublisher(
      const std::string &topic_name, const rclcpp::QoS &qos,
      const PublisherOptionsWithAllocator<AllocatorT> &options =
          PublisherOptionsWithAllocator<AllocatorT>());

  /// Create and return a Subscription.
  /**
   * \param[in] topic_name The topic to subscribe on.
   * \param[in] qos QoS profile for Subcription.
   * \param[in] callback The user-defined callback function to receive a message
   * \param[in] options Additional options for the creation of the Subscription.
   * \param[in] msg_mem_strat The message memory strategy to use for allocating
   * messages. \return Shared pointer to the created subscription.
   */
  template <typename MessageT, typename CallbackT,
            typename AllocatorT = std::allocator<void>,
            typename SubscriptionT = rclcpp::Subscription<MessageT, AllocatorT>,
            typename MessageMemoryStrategyT =
                typename SubscriptionT::MessageMemoryStrategyType>
  std::shared_ptr<SubscriptionT> CreateSubscription(
      const std::string &topic_name, const rclcpp::QoS &qos,
      CallbackT &&callback,
      const SubscriptionOptionsWithAllocator<AllocatorT> &options =
          SubscriptionOptionsWithAllocator<AllocatorT>(),
      typename MessageMemoryStrategyT::SharedPtr msg_mem_strat =
          (MessageMemoryStrategyT::create_default()));

  /// Create a timer.
  /**
   * \param[in] period Time interval between triggers of the callback.
   * \param[in] callback User-defined callback function.
   * \param[in] group Callback group to execute this timer's callback in.
   */
  template <typename DurationRepT = int64_t, typename DurationT = std::milli,
            typename CallbackT>
  typename rclcpp::WallTimer<CallbackT>::SharedPtr CreateWallTimer(
      std::chrono::duration<DurationRepT, DurationT> period, CallbackT callback,
      rclcpp::CallbackGroup::SharedPtr group = nullptr);

  /// Declare and initialize a parameter with a type.
  /**
   * See the non-templated DeclareParameter() on this class for details.
   *
   * If the type of the default value, and therefore also the type of return
   * value, differs from the initial value provided in the node options, then
   * a rclcpp::exceptions::InvalidParameterTypeException may be thrown.
   * To avoid this, use the DeclareParameter() method which returns an
   * rclcpp::ParameterValue instead.
   *
   * Note, this method cannot return a const reference, because extending the
   * lifetime of a temporary only works recursively with member initializers,
   * and cannot be extended to members of a class returned.
   * The return value of this class is a copy of the member of a ParameterValue
   * which is returned by the other version of DeclareParameter().
   * See also:
   *
   *   - https://en.cppreference.com/w/cpp/language/lifetime
   *   -
   * https://herbsutter.com/2008/01/01/gotw-88-a-candidate-for-the-most-important-const/
   *   - https://www.youtube.com/watch?v=uQyT-5iWUow (cppnow 2018 presentation)
   */
  template <typename ParameterT>
  auto DeclareParameter(
      const std::string &name, const ParameterT &default_value,
      const rcl_interfaces::msg::ParameterDescriptor &parameter_descriptor =
          rcl_interfaces::msg::ParameterDescriptor(),
      bool ignore_override = false);

  /// Get the parameter values for all parameters that have a given prefix.
  /**
   * The "prefix" argument is used to list the parameters which are prefixed
   * with that prefix, see also list_parameters().
   *
   * The resulting list of parameter names are used to get the values of the
   * parameters.
   *
   * The names which are used as keys in the values map have the prefix removed.
   * For example, if you use the prefix "foo" and the parameters "foo.ping" and
   * "foo.pong" exist, then the returned map will have the keys "ping" and
   * "pong".
   *
   * An empty string for the prefix will match all parameters.
   *
   * If no parameters with the prefix are found, then the output parameter
   * "values" will be unchanged and false will be returned.
   * Otherwise, the parameter names and values will be stored in the map and
   * true will be returned to indicate "values" was mutated.
   *
   * This method will never throw the
   * rclcpp::exceptions::ParameterNotDeclaredException exception because the
   * action of listing the parameters is done atomically with getting the
   * values, and therefore they are only listed if already declared and cannot
   * be undeclared before being retrieved.
   *
   * Like the templated get_parameter() variant, this method will attempt to
   * coerce the parameter values into the type requested by the given
   * template argument, which may fail and throw an exception.
   *
   * \param[in] prefix The prefix of the parameters to get.
   * \param[out] values The map used to store the parameter names and values,
   *   respectively, with one entry per parameter matching prefix.
   * \returns true if output "values" was changed, false otherwise.
   * \throws rclcpp::ParameterTypeException if the requested type does not
   *   match the value of the parameter which is stored.
   */
  template <typename ParameterT>
  bool GetParameters(const std::string &prefix,
                     std::map<std::string, ParameterT> &values) const;

  /// Add a callback for when parameters are being set.
  /**
   * The callback signature is designed to allow handling of any of the above
   * `set_parameter*` or `DeclareParameter*` methods, and so it takes a const
   * reference to a vector of parameters to be set, and returns an instance of
   * rcl_interfaces::msg::SetParametersResult to indicate whether or not the
   * parameter should be set or not, and if not why.
   *
   * For an example callback:
   *
   * ```cpp
   * rcl_interfaces::msg::SetParametersResult
   * my_callback(const std::vector<rclcpp::Parameter> &parameters)
   * {
   *   rcl_interfaces::msg::SetParametersResult result;
   *   result.successful = true;
   *   for (const auto &parameter : parameters) {
   *     if (!some_condition) {
   *       result.successful = false;
   *       result.reason = "the reason it could not be allowed";
   *     }
   *   }
   *   return result;
   * }
   * ```
   *
   * You can see that the SetParametersResult is a boolean flag for success
   * and an optional reason that can be used in error reporting when it fails.
   *
   * This allows the node developer to control which parameters may be changed.
   *
   * It is considered bad practice to reject changes for "unknown" parameters as
   * this prevents other parts of the node (that may be aware of these
   * parameters) from handling them.
   *
   * Note that the callback is called when DeclareParameter() and its variants
   * are called, and so you cannot assume the parameter has been set before
   * this callback, so when checking a new value against the existing one, you
   * must account for the case where the parameter is not yet set.
   *
   * Some constraints like read_only are enforced before the callback is called.
   *
   * The callback may introspect other already set parameters (by calling any
   * of the {get,list,describe}_parameter() methods), but may *not* modify
   * other parameters (by calling any of the {set,declare}_parameter() methods)
   * or modify the registered callback itself (by calling the
   * add_on_set_parameters_callback() method).  If a callback tries to do any
   * of the latter things,
   * rclcpp::exceptions::ParameterModifiedInCallbackException will be thrown.
   *
   * The callback functions must remain valid as long as the
   * returned smart pointer is valid.
   * The returned smart pointer can be promoted to a shared version.
   *
   * Resetting or letting the smart pointer go out of scope unregisters the
   * callback. `remove_on_set_parameters_callback` can also be used.
   *
   * The registered callbacks are called when a parameter is set.
   * When a callback returns a not successful result, the remaining callbacks
   * aren't called. The order of the callback is the reverse from the
   * registration order.
   *
   * \param callback The callback to register.
   * \returns A shared pointer. The callback is valid as long as the smart
   * pointer is alive. \throws std::bad_alloc if the allocation of the
   * OnSetParametersCallbackHandle fails.
   */
  RCLCPP_PUBLIC
  RCUTILS_WARN_UNUSED
  OnSetParametersCallbackHandle::SharedPtr DddOnSetParametersCallback(
      OnParametersSetCallbackType callback);

 protected:
  /// Construct a sub-node, which will extend the namespace of all entities
  /// created with it.
  /**
   * \sa create_sub_node()
   *
   * \param[in] other The node from which a new sub-node is created.
   * \param[in] sub_namespace The sub-namespace of the sub-node.
   */
  RCLCPP_PUBLIC
  Node(const Node &other, const std::string &sub_namespace);

 private:
  RCLCPP_DISABLE_COPY(Node)

  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr node_base_;
  rclcpp::node_interfaces::NodeGraphInterface::SharedPtr node_graph_;
  rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr node_logging_;
  rclcpp::node_interfaces::NodeTimersInterface::SharedPtr node_timers_;
  rclcpp::node_interfaces::NodeTopicsInterface::SharedPtr node_topics_;
  rclcpp::node_interfaces::NodeServicesInterface::SharedPtr node_services_;
  rclcpp::node_interfaces::NodeClockInterface::SharedPtr node_clock_;
  rclcpp::node_interfaces::NodeParametersInterface::SharedPtr node_parameters_;
  rclcpp::node_interfaces::NodeTimeSourceInterface::SharedPtr node_time_source_;
  rclcpp::node_interfaces::NodeWaitablesInterface::SharedPtr node_waitables_;

  const rclcpp::NodeOptions node_options_;
  const std::string sub_namespace_;
  const std::string effective_namespace_;
};

RCLCPP_LOCAL
inline std::string ExtendNameWithSubNamespace(
    const std::string &name, const std::string &sub_namespace) {
  std::string name_with_sub_namespace(name);
  if (sub_namespace != "" && name.front() != '/' && name.front() != '~') {
    name_with_sub_namespace = sub_namespace + "/" + name;
  }
  return name_with_sub_namespace;
}

template <typename MessageT, typename AllocatorT, typename PublisherT>
std::shared_ptr<PublisherT> Node::CreatePublisher(
    const std::string &topic_name, const rclcpp::QoS &qos,
    const PublisherOptionsWithAllocator<AllocatorT> &options) {
  return rclcpp::create_publisher<MessageT, AllocatorT, PublisherT>(
      *this,
      extend_name_with_sub_namespace(topic_name, this->get_sub_namespace()),
      qos, options);
}

template <typename MessageT, typename CallbackT, typename AllocatorT,
          typename SubscriptionT, typename MessageMemoryStrategyT>
std::shared_ptr<SubscriptionT> Node::CreateSubscription(
    const std::string &topic_name, const rclcpp::QoS &qos, CallbackT &&callback,
    const SubscriptionOptionsWithAllocator<AllocatorT> &options,
    typename MessageMemoryStrategyT::SharedPtr msg_mem_strat) {
  return rclcpp::create_subscription<MessageT>(
      *this,
      extend_name_with_sub_namespace(topic_name, this->get_sub_namespace()),
      qos, std::forward<CallbackT>(callback), options, msg_mem_strat);
}

template <typename DurationRepT, typename DurationT, typename CallbackT>
typename rclcpp::WallTimer<CallbackT>::SharedPtr Node::CreateWallTimer(
    std::chrono::duration<DurationRepT, DurationT> period, CallbackT callback,
    rclcpp::CallbackGroup::SharedPtr group) {
  return rclcpp::create_wall_timer(period, std::move(callback), group,
                                   this->node_base_.get(),
                                   this->node_timers_.get());
}

template <typename ParameterT>
auto Node::DeclareParameter(
    const std::string &name,
    const rcl_interfaces::msg::ParameterDescriptor &parameter_descriptor,
    bool ignore_override) {
  // get advantage of parameter value template magic to get
  // the correct rclcpp::ParameterType from ParameterT
  rclcpp::ParameterValue value{ParameterT{}};
  return this
      ->DeclareParameter(name, value.get_type(), parameter_descriptor,
                         ignore_override)
      .get<ParameterT>();
}

// this is a partially-specialized version of get_parameter above,
// where our concrete type for ParameterT is std::map, but the to-be-determined
// type is the value in the map.
template <typename ParameterT>
bool Node::GetParameters(const std::string &prefix,
                         std::map<std::string, ParameterT> &values) const {
  std::map<std::string, rclcpp::Parameter> params;
  bool result = node_parameters_->get_parameters_by_prefix(prefix, params);
  if (result) {
    for (const auto &param : params) {
      values[param.first] =
          static_cast<ParameterT>(param.second.get_value<ParameterT>());
    }
  }

  return result;
}

}  // namespace rclcpp

#endif  // RCLCPP__NODE_IMPL_HPP_
```
